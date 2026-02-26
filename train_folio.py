#%%
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)
#%%
total_training_steps = 12_000
batch_size = 1024
total_training_tokens = total_training_steps * batch_size
#%%
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    model_name="Qwen/Qwen3-8B",
    hook_name="blocks.20.hook_mlp_out",
    dataset_path="/mount/arbeitsdaten66/projekte/multiview/hardy/datasets/FOLIO_tokenized/",
    is_dataset_tokenized=False,
    streaming=False,

    # SAE Parameters are in the nested 'sae' config
    sae=StandardTrainingSAEConfig(
        d_in=4096, 
        d_sae=8192,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
    ),

    # Training Parameters
    lr=5e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    train_batch_size_tokens=batch_size,

    # Activation Store Parameters
    context_size=128,
    n_batches_in_buffer=8,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=2,

    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="test-folio",
        wandb_log_frequency=100,
        eval_every_n_wandb_logs=20,
    ),

    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="bfloat16"
)
#%%
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
sparse_autoencoder.save_inference_model()