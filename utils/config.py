"""
Configuration file for RNN and Transformer language models.
Optimized for Mac (MPS/CPU) training.
"""

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration for the project."""

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_wikipedia_dir: Path = project_root / "data" / "wikipedia"
    checkpoints_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    # Device configuration (Mac-optimized)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data processing
    vocab_size: int = 10000  # Small vocab for Mac training
    max_seq_length: int = 128  # Shorter sequences for faster training
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05

    # Tokenizer
    min_frequency: int = 2  # Minimum token frequency to include in vocab

    # Training hyperparameters
    batch_size: int = 32  # Small batch size for Mac
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Model saving
    save_every_n_epochs: int = 2

    # Plotting
    plot_every_n_epochs: int = 1  # Generate plots after every N epochs

    # Evaluation
    eval_batch_size: int = 64

    # Generation
    max_gen_length: int = 100
    temperature: float = 1.0
    top_k: int = 50

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RNNConfig(Config):
    """Configuration specific to RNN model."""

    # Model architecture
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    rnn_type: str = "LSTM"  # Options: LSTM, GRU

    # Estimated parameters: ~5-8M

    def __repr__(self):
        return (f"RNN({self.rnn_type}, layers={self.num_layers}, "
                f"emb={self.embedding_dim}, hidden={self.hidden_dim})")


@dataclass
class TransformerConfig(Config):
    """Configuration specific to Transformer model."""

    # Model architecture
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024  # Feed-forward dimension
    dropout: float = 0.1

    # Estimated parameters: ~8-12M

    def __repr__(self):
        return (f"Transformer(layers={self.num_layers}, heads={self.num_heads}, "
                f"emb={self.embedding_dim}, ff={self.ff_dim})")


def get_config(model_type: str = "rnn") -> Config:
    """
    Get configuration for specified model type.

    Args:
        model_type: Either 'rnn' or 'transformer'

    Returns:
        Configuration object
    """
    if model_type.lower() == "rnn":
        return RNNConfig()
    elif model_type.lower() == "transformer":
        return TransformerConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test configurations
    rnn_config = get_config("rnn")
    transformer_config = get_config("transformer")

    print("RNN Config:")
    print(f"  Device: {rnn_config.device}")
    print(f"  Vocab size: {rnn_config.vocab_size}")
    print(f"  Model: {rnn_config}")
    print()
    print("Transformer Config:")
    print(f"  Device: {transformer_config.device}")
    print(f"  Vocab size: {transformer_config.vocab_size}")
    print(f"  Model: {transformer_config}")
