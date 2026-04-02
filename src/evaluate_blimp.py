"""
Evaluate the model on the BLIMP dataset
"""
#%%
import argparse
import tqdm
import json
import torch
from pathlib import Path
from util.blimp_util import build_from_blimp_row
from transformers import AutoTokenizer, GPT2LMHeadModel


#%%
tokenizer = AutoTokenizer.from_pretrained('gpt2-large', device='cuda')
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Using device: {device}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def evaluate_sentence(batch_data):
    batch_text = []
    for data in batch_data:
        # Collecting the good and bad sentences in the batch
        # Batch size would be doubled
        batch_text.append(data[0]['prompt'])
        batch_text.append(data[1]['prompt'])
    single_input = isinstance(batch_text, str)
    texts = [batch_text] if single_input else batch_text

    enc = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)

    logits = out.logits  # [B, T, V]
    shift_logits = logits[:, :-1, :]      # predict token t+1 from prefix up to t
    shift_labels = input_ids[:, 1:]       # gold next tokens
    shift_mask = attention_mask[:, 1:]     # valid target positions only

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    token_logp = token_logp * shift_mask
    sent_loglik = token_logp.sum(dim=1)   # [B]

    if single_input:
        return sent_loglik[0].item()
    return sent_loglik.tolist()

def evaluate_batch(batch_data):
    
    batch_loglikes = evaluate_sentence(batch_data)
    for i, data in enumerate(batch_data):
        good = batch_loglikes[i * 2]
        bad = batch_loglikes[i * 2 + 1]
        if good > bad:
            data[0]['prediction'] = 'good'
            data[1]['prediction'] = 'bad'
        else:
            data[0]['prediction'] = 'bad'
            data[1]['prediction'] = 'good'
#%%

def main(args):
    # Read the dataset 
    blimp_path = Path(args.blimp_path)
    # BLIMP jsonl files are grouped into different linguistic phenomena
    blimp_files = list(blimp_path.glob("*.jsonl")) 
    blimp_data = []
    for file in tqdm.tqdm(blimp_files):
        with file.open() as f:
            blimp =  [json.loads(line) for line in f]
            blimp_data.extend([build_from_blimp_row(row) for row in blimp])
    
    all_lines = []
    correct = 0
    batch = args.batch_size
    for i in tqdm.tqdm(range(0, len(blimp_data), batch)):
        batch_data = blimp_data[i:i+batch]
        evaluate_batch(batch_data)        
        for data in batch_data:
            if data[0]['label'] == data[0]['prediction']:
                correct += 1
    accuracy = correct / len(blimp_data)
    print(f"Accuracy: {accuracy:.4f}")
    for data in blimp_data:
        jsonline = json.dumps(data)
        all_lines.append(jsonline)
    with open(args.output_path, 'w') as f:
        f.write('\n'.join(all_lines))

# Save each sample result into a new file
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Evaluate the model on the BLIMP set')
    args.add_argument('--batch_size', type=int, default=15, help='Batch size for evaluation')
    args.add_argument('--blimp_path', type=str, default='data/blimp', help='Path to the BLIMP dataset')
    args.add_argument('--output_path', type=str, default='blimp_results.jsonl', help='Path to save the results')
    args = args.parse_args()
    main(args)