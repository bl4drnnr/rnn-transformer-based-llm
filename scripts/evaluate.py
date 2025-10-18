"""
Evaluation script for trained language models.
Evaluates on test data and calculates perplexity.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.dataset import create_dataloaders
from utils.metrics import evaluate_model
from models.rnn_model import RNNLanguageModel
from models.transformer_model import TransformerLanguageModel


def evaluate(
    model_type: str,
    checkpoint_path: str,
    data_type: str = "test",
):
    """
    Evaluate a trained model.

    Args:
        model_type: Type of model ('rnn' or 'transformer')
        checkpoint_path: Path to model checkpoint
        data_type: Type of data to evaluate on ('test', 'val', or 'wikipedia')
    """
    print("=" * 80)
    print(f"EVALUATING {model_type.upper()} LANGUAGE MODEL")
    print("=" * 80)

    # Load configuration
    config = get_config(model_type)
    print(f"\nConfiguration: {config}")
    print(f"Device: {config.device}")

    # Load metadata
    with open(config.data_processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
        pad_token_id = metadata["pad_token_id"]

    # Load data based on type
    print(f"\nLoading {data_type} data...")
    if data_type == "test":
        eval_ids = torch.load(config.data_processed_dir / "test_ids.pt")
    elif data_type == "val":
        eval_ids = torch.load(config.data_processed_dir / "val_ids.pt")
    elif data_type == "wikipedia":
        wiki_path = config.data_processed_dir / "wikipedia_ids.pt"
        if not wiki_path.exists():
            print(f"Error: Wikipedia data not found at {wiki_path}")
            print("Please preprocess Wikipedia data first.")
            return
        eval_ids = torch.load(wiki_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    print(f"Loaded {len(eval_ids)} sequences")

    # Create dataloader
    from torch.utils.data import DataLoader
    from utils.dataset import LanguageModelingDataset

    eval_dataset = LanguageModelingDataset(
        eval_ids,
        max_length=config.max_seq_length,
        pad_token_id=pad_token_id,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # Create model
    print("Initializing model...")
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Evaluate
    print(f"\nEvaluating on {data_type} set...")
    criterion = nn.CrossEntropyLoss(reduction='mean')

    start_time = time.time()
    loss, perplexity = evaluate_model(
        model,
        eval_loader,
        criterion,
        config.device,
        pad_token_id,
    )
    eval_time = time.time() - start_time

    # Calculate tokens per second
    total_tokens = sum(len(seq) for seq in eval_ids)
    tokens_per_second = total_tokens / eval_time

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {model_type.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_type}")
    print(f"Sequences: {len(eval_ids)}")
    print(f"Total tokens: {total_tokens:,}")
    print("-" * 80)
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"Throughput: {tokens_per_second:.0f} tokens/s")
    print("=" * 80)

    # Save results
    results = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "data_type": data_type,
        "num_sequences": len(eval_ids),
        "total_tokens": total_tokens,
        "loss": loss,
        "perplexity": perplexity,
        "eval_time": eval_time,
        "tokens_per_second": tokens_per_second,
    }

    results_path = config.results_dir / f"{model_type}_eval_{data_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained language model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rnn", "transformer"],
        help="Model type",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        choices=["test", "val", "wikipedia"],
        help="Data to evaluate on",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    evaluate(args.model, str(checkpoint_path), args.data)


if __name__ == "__main__":
    main()
