"""
Training script for RNN and Transformer language models.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import time
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config, RNNConfig, TransformerConfig
from utils.dataset import create_dataloaders
from utils.metrics import MetricsTracker, evaluate_model, calculate_perplexity
from utils.plotting import TrainingPlotter
from models.rnn_model import RNNLanguageModel
from models.transformer_model import TransformerLanguageModel


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    pad_token_id: int,
    gradient_clip: float = 1.0,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss (ignore padding tokens)
        batch_size, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        mask = (targets_flat != pad_token_id)
        loss = criterion(outputs_flat[mask], targets_flat[mask])

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Update weights
        optimizer.step()

        # Track statistics
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss


def train(
    model_type: str = "rnn",
    resume_from: str = None,
):
    """
    Main training function.

    Args:
        model_type: Type of model ('rnn' or 'transformer')
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 80)
    print(f"TRAINING {model_type.upper()} LANGUAGE MODEL")
    print("=" * 80)

    # Load configuration
    config = get_config(model_type)
    print(f"\nConfiguration: {config}")
    print(f"Device: {config.device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    train_ids = torch.load(config.data_processed_dir / "train_ids.pt")
    val_ids = torch.load(config.data_processed_dir / "val_ids.pt")
    test_ids = torch.load(config.data_processed_dir / "test_ids.pt")

    with open(config.data_processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
        pad_token_id = metadata["pad_token_id"]

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ids,
        val_ids,
        test_ids,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        pad_token_id=pad_token_id,
        num_workers=0,  # Mac-friendly
    )

    # Create model
    print("\nInitializing model...")
    if model_type == "rnn":
        model = RNNLanguageModel(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            rnn_type=config.rnn_type,
            pad_token_id=pad_token_id,
        )
    else:  # transformer
        model = TransformerLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.ff_dim,
            max_seq_len=config.max_seq_length,
            dropout=config.dropout,
            pad_token_id=pad_token_id,
        )

    model = model.to(config.device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )

    # Metrics tracker
    metrics = MetricsTracker()

    # Initialize plotter
    plotter = TrainingPlotter(save_dir=config.results_dir, model_name=model_type)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    total_training_time = 0.0

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 80)

        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
            pad_token_id,
            config.gradient_clip,
        )

        # Validate
        print("Validating...")
        val_loss, val_ppl = evaluate_model(
            model,
            val_loader,
            criterion,
            config.device,
            pad_token_id,
        )

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        # Update learning rate
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Track metrics
        metrics.update(train_loss, val_loss, current_lr, epoch_time)

        # Print epoch summary
        train_ppl = calculate_perplexity(train_loss)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or val_loss < best_val_loss:
            checkpoint_path = config.checkpoints_dir / f"{model_type}_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = config.checkpoints_dir / f"{model_type}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, best_model_path)
            print(f"  Best model saved: {best_model_path}")

        # Generate and save plots
        plotter.plot_all_metrics(metrics, epoch, plot_frequency=config.plot_every_n_epochs)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {total_training_time:.1f}s ({total_training_time / 60:.1f}m)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")

    # Save metrics
    metrics_path = config.results_dir / f"{model_type}_metrics.json"
    metrics.save(str(metrics_path))

    # Create final summary plot
    plotter.create_final_summary_plot(metrics_path)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Training plots saved to: {plotter.plots_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument(
        "--model",
        type=str,
        default="rnn",
        choices=["rnn", "transformer"],
        help="Model type to train",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    train(model_type=args.model, resume_from=args.resume)


if __name__ == "__main__":
    main()
