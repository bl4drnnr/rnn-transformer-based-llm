"""
Visualize training metrics from saved JSON files.
Useful for creating plots after training or comparing different runs.
"""

import sys
from pathlib import Path
import argparse
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.plotting import TrainingPlotter


def visualize_metrics(metrics_path: str, model_name: str = None):
    """
    Create plots from saved metrics JSON file.

    Args:
        metrics_path: Path to metrics JSON file
        model_name: Optional name override for plots
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return

    # Determine model name from filename if not provided
    if model_name is None:
        model_name = metrics_path.stem.replace('_metrics', '')

    print(f"Visualizing metrics from: {metrics_path}")
    print(f"Model name: {model_name}")

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create plotter
    save_dir = metrics_path.parent
    plotter = TrainingPlotter(save_dir=save_dir, model_name=model_name)

    # Extract data
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_ppls = metrics['train_perplexities']
    val_ppls = metrics['val_perplexities']
    lrs = metrics['learning_rates']
    times = metrics['epoch_times']
    final_epoch = len(train_losses)

    print(f"\nGenerating plots for {final_epoch} epochs...")

    # Create all plots
    plotter.plot_losses(train_losses, val_losses, final_epoch)
    plotter.plot_perplexity(train_ppls, val_ppls, final_epoch)
    plotter.plot_learning_rate(lrs, final_epoch)
    plotter.plot_epoch_times(times, final_epoch)
    plotter.plot_combined_metrics(
        train_losses, val_losses,
        train_ppls, val_ppls,
        lrs, times,
        final_epoch
    )
    plotter.plot_loss_comparison(train_losses, val_losses, final_epoch)

    # Create summary plot
    plotter._plot_summary_stats(metrics)

    print(f"\nPlots saved to: {plotter.plots_dir}")
    print("\nGenerated plots:")
    print(f"  - {model_name}_loss_epoch_{final_epoch}.png")
    print(f"  - {model_name}_perplexity_epoch_{final_epoch}.png")
    print(f"  - {model_name}_learning_rate_epoch_{final_epoch}.png")
    print(f"  - {model_name}_epoch_times_epoch_{final_epoch}.png")
    print(f"  - {model_name}_combined_metrics_epoch_{final_epoch}.png")
    print(f"  - {model_name}_overfitting_analysis_epoch_{final_epoch}.png")
    print(f"  - {model_name}_summary_stats.png")


def compare_models(rnn_metrics_path: str, transformer_metrics_path: str):
    """
    Create comparison plots between RNN and Transformer models.

    Args:
        rnn_metrics_path: Path to RNN metrics JSON
        transformer_metrics_path: Path to Transformer metrics JSON
    """
    import matplotlib.pyplot as plt

    rnn_path = Path(rnn_metrics_path)
    transformer_path = Path(transformer_metrics_path)

    if not rnn_path.exists() or not transformer_path.exists():
        print("Error: One or both metrics files not found")
        return

    print(f"Comparing models:")
    print(f"  RNN: {rnn_path}")
    print(f"  Transformer: {transformer_path}")

    # Load both metrics
    with open(rnn_path, 'r') as f:
        rnn_metrics = json.load(f)
    with open(transformer_path, 'r') as f:
        transformer_metrics = json.load(f)

    # Create comparison plots
    save_dir = rnn_path.parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs_rnn = range(1, len(rnn_metrics['train_losses']) + 1)
    epochs_transformer = range(1, len(transformer_metrics['train_losses']) + 1)

    # Training loss
    ax1.plot(epochs_rnn, rnn_metrics['train_losses'], 'b-o', label='RNN Train', linewidth=2, markersize=5)
    ax1.plot(epochs_rnn, rnn_metrics['val_losses'], 'b--s', label='RNN Val', linewidth=2, markersize=5)
    ax1.plot(epochs_transformer, transformer_metrics['train_losses'], 'r-o', label='Transformer Train', linewidth=2, markersize=5)
    ax1.plot(epochs_transformer, transformer_metrics['val_losses'], 'r--s', label='Transformer Val', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Perplexity
    ax2.plot(epochs_rnn, rnn_metrics['val_perplexities'], 'b-o', label='RNN', linewidth=2, markersize=6)
    ax2.plot(epochs_transformer, transformer_metrics['val_perplexities'], 'r-s', label='Transformer', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Perplexity Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('RNN vs Transformer - Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plot saved to: {save_path}")

    # Print summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'RNN':<20} {'Transformer':<20}")
    print("-" * 80)
    print(f"{'Best Val Loss':<30} {rnn_metrics['best_val_loss']:<20.4f} {transformer_metrics['best_val_loss']:<20.4f}")
    print(f"{'Best Val Perplexity':<30} {rnn_metrics['best_val_perplexity']:<20.2f} {transformer_metrics['best_val_perplexity']:<20.2f}")
    print(f"{'Total Training Time (s)':<30} {rnn_metrics['total_time']:<20.1f} {transformer_metrics['total_time']:<20.1f}")
    print(f"{'Avg Epoch Time (s)':<30} {rnn_metrics['avg_epoch_time']:<20.1f} {transformer_metrics['avg_epoch_time']:<20.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for plot titles (optional)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare RNN and Transformer models",
    )
    parser.add_argument(
        "--rnn-metrics",
        type=str,
        default="results/rnn_metrics.json",
        help="Path to RNN metrics (for comparison)",
    )
    parser.add_argument(
        "--transformer-metrics",
        type=str,
        default="results/transformer_metrics.json",
        help="Path to Transformer metrics (for comparison)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(args.rnn_metrics, args.transformer_metrics)
    elif args.metrics:
        visualize_metrics(args.metrics, args.model_name)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/visualize_metrics.py --metrics results/rnn_metrics.json")
        print("  python scripts/visualize_metrics.py --compare")


if __name__ == "__main__":
    main()
