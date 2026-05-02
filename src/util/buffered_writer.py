import torch
import h5py
import numpy as np

class BufferedFeatureWriter:
    """
    Buffers feature activations in memory and flushes to HDF5 in large chunks.
    Pre-allocates datasets with a capacity estimate to avoid repeated resizing.
    """
    def __init__(self, h5py_f, n_sentences: int, n_features: int, capacity_multiplier: float = 0.1, overwrite: bool = False):
        self.f = h5py_f
        self.n_sentences = n_sentences

        if not overwrite and (self.f.keys() or self.f.attrs):
            raise ValueError("HDF5 file is not empty. Set overwrite=True to clear it.")
        # If the file handle already contains datasets/attributes, clear it so
        # this writer always starts from a clean state.
        if self.f.keys() or self.f.attrs:
            for key in list(self.f.keys()):
                del self.f[key]
            for attr_key in list(self.f.attrs.keys()):
                del self.f.attrs[attr_key]

        # --- Pre-allocated sparse storage ---
        # Estimate nnz capacity: assume `capacity_multiplier` fraction of all (token, feature) pairs are active
        # Resize if needed, but ideally this is set generously enough to avoid it.
        estimated_nnz = int(n_sentences * 128 * n_features * capacity_multiplier)  # 128 = avg token length guess

        self.ds_sentence_idx  = h5py_f.create_dataset("sentence_idx",  shape=(estimated_nnz,), maxshape=(None,), dtype="int32",   chunks=(65536,))
        self.ds_token_idx     = h5py_f.create_dataset("token_idx",     shape=(estimated_nnz,), maxshape=(None,), dtype="int32",   chunks=(65536,))
        self.ds_feature_idx   = h5py_f.create_dataset("feature_idx",   shape=(estimated_nnz,), maxshape=(None,), dtype="int32",   chunks=(65536,))
        self.ds_feature_vals  = h5py_f.create_dataset("feature_values",shape=(estimated_nnz,), maxshape=(None,), dtype="float32", chunks=(65536,))

        # Sentence-level metadata (one row per sentence)
        self.ds_tokens        = h5py_f.create_dataset("tokens",        shape=(n_sentences,),   maxshape=(None,), dtype=h5py.special_dtype(vlen=np.int32))
        self.ds_offsets       = h5py_f.create_dataset("offsets",       shape=(n_sentences + 1,), dtype="int64")  # CSR-style: offsets[i]:offsets[i+1] = nnz range for sentence i

        h5py_f.attrs["n_features"] = n_features

        # --- In-memory buffer ---
        self.buf_sentence_idx  = []
        self.buf_token_idx     = []
        self.buf_feature_idx   = []
        self.buf_feature_vals  = []

        self.flush_every  = 100_000   # flush to disk when buffer exceeds this many entries
        self.nnz_cursor   = 0         # how many entries have been written to HDF5 so far
        self.sent_cursor  = 0         # how many sentences have been finalized

    def add_batch(self, feature_acts: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_start_idx: int):
        """
        feature_acts: [B, T, F] on GPU
        input_ids:    [B, T]    on GPU  (already includes BOS)
        attention_mask: [B, T]  on GPU
        batch_start_idx: global sentence index for feature_acts[0]
        """
        # Do all indexing on GPU, transfer the sparse result once per batch
        B = feature_acts.shape[0]

        for b in range(B):
            mask = attention_mask[b].bool()                     # [T]
            non_pad_acts  = feature_acts[b][mask]               # [T', F]
            non_pad_tokens = input_ids[b][mask]                 # [T']

            tok_idx, feat_idx = torch.nonzero(non_pad_acts, as_tuple=True)
            feat_vals = non_pad_acts[tok_idx, feat_idx]

            global_sent_idx = batch_start_idx + b

            # Single GPU→CPU transfer per sentence (sparse only)
            self.buf_sentence_idx.append(np.full(len(tok_idx), global_sent_idx, dtype=np.int32))
            self.buf_token_idx.append(tok_idx.cpu().numpy().astype(np.int32))
            self.buf_feature_idx.append(feat_idx.cpu().numpy().astype(np.int32))
            self.buf_feature_vals.append(feat_vals.cpu().numpy().astype(np.float32))

            # Store variable-length token array directly (one write per sentence, small)
            self.ds_tokens[global_sent_idx] = non_pad_tokens.cpu().numpy().astype(np.int32)

        # Flush if buffer is large enough
        total_buffered = sum(len(x) for x in self.buf_token_idx)
        if total_buffered >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self.buf_token_idx:
            return

        sent  = np.concatenate(self.buf_sentence_idx)
        tok   = np.concatenate(self.buf_token_idx)
        feat  = np.concatenate(self.buf_feature_idx)
        vals  = np.concatenate(self.buf_feature_vals)
        n     = len(tok)

        end = self.nnz_cursor + n

        # Resize if we underestimated capacity
        if end > self.ds_token_idx.shape[0]:
            new_size = int(end * 1.5)
            for ds in (self.ds_sentence_idx, self.ds_token_idx, self.ds_feature_idx, self.ds_feature_vals):
                ds.resize(new_size, axis=0)

        self.ds_sentence_idx [self.nnz_cursor:end] = sent
        self.ds_token_idx    [self.nnz_cursor:end] = tok
        self.ds_feature_idx  [self.nnz_cursor:end] = feat
        self.ds_feature_vals [self.nnz_cursor:end] = vals

        self.nnz_cursor = end

        self.buf_sentence_idx.clear()
        self.buf_token_idx.clear()
        self.buf_feature_idx.clear()
        self.buf_feature_vals.clear()

    def finalize(self):
        """Call once after all batches. Flushes remainder and trims datasets to actual size."""
        self._flush()

        # Trim pre-allocated datasets to actual nnz
        for ds in (self.ds_sentence_idx, self.ds_token_idx, self.ds_feature_idx, self.ds_feature_vals):
            ds.resize(self.nnz_cursor, axis=0)

        # Write CSR-style offsets so callers can slice by sentence without loading all data
        # offsets[i] = start of sentence i in the sparse arrays
        # offsets[n] = total nnz
        offsets = np.zeros(self.n_sentences + 1, dtype=np.int64)
        sent_counts = np.bincount(
            self.ds_sentence_idx[:],     # read back the finalized array
            minlength=self.n_sentences
        )
        np.cumsum(sent_counts, out=offsets[1:])
        self.ds_offsets[:] = offsets

        self.f.attrs["n_sentences"] = self.n_sentences
        self.f.attrs["total_nnz"]   = self.nnz_cursor