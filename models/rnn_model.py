"""
RNN-based language model (LSTM/GRU) for causal language modeling.
"""

import torch
import torch.nn as nn
from typing import Optional


class RNNLanguageModel(nn.Module):
    """
    LSTM/GRU-based language model for next token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "LSTM",
        pad_token_id: int = 0,
    ):
        """
        Initialize RNN language model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
            rnn_type: Type of RNN ('LSTM' or 'GRU')
            pad_token_id: ID of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.pad_token_id = pad_token_id

        # Token embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_token_id
        )

        # RNN layer (LSTM or GRU)
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=0.1)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            hidden: Initial hidden state (optional)

        Returns:
            logits: Predictions for next token (batch_size, seq_len, vocab_size)
        """
        # Embed tokens: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Pass through RNN: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        if hidden is not None:
            rnn_output, _ = self.rnn(embedded, hidden)
        else:
            rnn_output, _ = self.rnn(embedded)

        # Apply dropout
        rnn_output = self.dropout(rnn_output)

        # Project to vocabulary: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.output_projection(rnn_output)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs (batch_size, initial_seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample from top-k tokens only
            eos_token_id: End-of-sequence token ID (stops generation)

        Returns:
            generated_ids: Generated token IDs (batch_size, final_seq_len)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits = self.forward(generated)  # (batch_size, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if EOS token is generated (for all sequences in batch)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test RNN model
    vocab_size = 10000
    batch_size = 4
    seq_len = 32

    model = RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        rnn_type="LSTM",
    )

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"Model: {model.rnn_type}-based Language Model")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")

    # Test generation
    initial_ids = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(initial_ids, max_length=20, temperature=1.0)
    print(f"\nGeneration test:")
    print(f"Initial length: {initial_ids.size(1)}")
    print(f"Generated length: {generated.size(1)}")
