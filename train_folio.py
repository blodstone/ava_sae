"""
Training script for a Sparse Autoencoder (SAE) on the FOLIO dataset using Qwen3-8B model.

Configures and trains an SAE to learn interpretable features from the model's MLP output layer,
with monitoring via Weights & Biases.
"""
import argparse

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)


def build_runner_config(dataset_path: str) -> LanguageModelSAERunnerConfig:
    total_training_steps = 12_000
    batch_size = 1024
    total_training_tokens = total_training_steps * batch_size
    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distribution)
        model_name="Qwen/Qwen3-8B",
        hook_name="blocks.20.hook_mlp_out",
        dataset_path=dataset_path,
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
        dtype="bfloat16",
    )


def train_and_save(cfg: LanguageModelSAERunnerConfig) -> None:
    sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()
    sparse_autoencoder.save_inference_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on a tokenized FOLIO dataset.")
    parser.add_argument("input_file", help="Path to the tokenized FOLIO dataset")
    args = parser.parse_args()

    cfg = build_runner_config(args.input_file)
    train_and_save(cfg)


if __name__ == "__main__":
    main()