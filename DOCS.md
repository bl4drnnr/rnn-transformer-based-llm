# RNN vs Transformer Language Models - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architectures](#model-architectures)
   - [RNN (LSTM) Architecture](#rnn-lstm-architecture)
   - [Transformer Architecture](#transformer-architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Complete Workflow](#complete-workflow)
6. [Detailed Usage Guide](#detailed-usage-guide)
7. [Configuration](#configuration)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Tips & Troubleshooting](#tips--troubleshooting)
10. [Expected Results](#expected-results)

---

## Project Overview

This project implements and compares two neural language model architectures for causal language modeling (next token prediction) on Polish text:

1. **RNN-based model** using LSTM (Long Short-Term Memory)
2. **Transformer-based model** using multi-head self-attention

### Key Features

- **Causal Language Modeling** - Predicts the next token in a sequence
- **BPE Tokenization** - Custom Byte-Pair Encoding tokenizer trained on your data
- **Dual Evaluation** - In-domain (test set) and out-of-domain (Wikipedia)
- **Perplexity Metrics** - Standard metric for language model quality
- **Time Tracking** - Training and inference time measurement
- **Text Generation** - Autoregressive text generation with temperature and top-k sampling
- **Mac Optimized** - MPS (Metal Performance Shaders) support for Apple Silicon

### Lab Assignment Context

This implementation fulfills the requirements of the Computational Linguistics lab assignment:
- Train two different architectures (RNN and Transformer)
- Evaluate using perplexity on held-out data
- Measure and compare training/inference time
- Generate text completions for at least 10 prompts
- Compare in-domain vs out-of-domain performance

---

## Model Architectures

### RNN (LSTM) Architecture

The RNN model uses LSTM cells for sequential processing. Here's how it works:

```
INPUT SEQUENCE
    |
    v
+-------------------------------------------------------------+
|  TOKEN EMBEDDING LAYER                                      |
|  Maps token IDs to dense vectors                            |
|  Input:  (batch_size, seq_len)                             |
|  Output: (batch_size, seq_len, embedding_dim=256)          |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  DROPOUT (p=0.3)                                            |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  LSTM LAYER 1                                               |
|  +-----------+  +-----------+  +-----------+                |
|  |  LSTM     |->|  LSTM     |->|  LSTM     | ...            |
|  |  Cell     |  |  Cell     |  |  Cell     |                |
|  |  t=1      |  |  t=2      |  |  t=3      |                |
|  +-----------+  +-----------+  +-----------+                |
|  Hidden dim: 512                                            |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  LSTM LAYER 2                                               |
|  +-----------+  +-----------+  +-----------+                |
|  |  LSTM     |->|  LSTM     |->|  LSTM     | ...            |
|  |  Cell     |  |  Cell     |  |  Cell     |                |
|  |  t=1      |  |  t=2      |  |  t=3      |                |
|  +-----------+  +-----------+  +-----------+                |
|  Hidden dim: 512                                            |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  DROPOUT (p=0.3)                                            |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  LINEAR PROJECTION                                          |
|  Projects to vocabulary size                                |
|  Input:  (batch_size, seq_len, hidden_dim=512)            |
|  Output: (batch_size, seq_len, vocab_size=10000)          |
+-------------------------------------------------------------+
    |
    v
LOGITS (predictions for next token)
```

#### LSTM Cell Detail

Each LSTM cell contains:
```
       +------------------------------------+
       |         LSTM CELL                  |
       |                                    |
   +---|  Input Gate:  i_t = sigma(W_i*x_t)   |
   |   |  Forget Gate: f_t = sigma(W_f*x_t)   |
h_t|   |  Cell Gate:   g_t = tanh(W_g*x_t)|
   |   |  Output Gate: o_t = sigma(W_o*x_t)   |
   |   |                                    |
   |   |  Cell State: c_t = f_t * c_{t-1}   |
   |   |                    + i_t * g_t      |
   |   |  Hidden:     h_t = o_t * tanh(c_t) |
   +---|                                    |
       +------------------------------------+
```

#### RNN Model Parameters
- **Embedding dimension**: 256
- **Hidden dimension**: 512
- **Number of layers**: 2
- **Dropout**: 0.3
- **Total parameters**: ~5-8M

**Key Characteristics:**
- Sequential processing (one token at a time)
- Maintains hidden state across sequence
- Good at capturing local dependencies
- Can struggle with very long-range dependencies
- Faster training for short sequences

---

### Transformer Architecture

The Transformer model uses self-attention mechanisms. Based on "Attention Is All You Need" (Vaswani et al., 2017).

```
INPUT SEQUENCE
    |
    v
+-------------------------------------------------------------+
|  TOKEN EMBEDDING LAYER                                      |
|  Maps token IDs to dense vectors                            |
|  Input:  (batch_size, seq_len)                             |
|  Output: (batch_size, seq_len, d_model=256)                |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  SCALE EMBEDDINGS                                           |
|  Multiply by sqrt(d_model)                                  |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  POSITIONAL ENCODING                                        |
|  Add sinusoidal position information                        |
|  PE(pos,2i)   = sin(pos / 10000^(2i/d_model))             |
|  PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))             |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  DROPOUT (p=0.1)                                            |
+-------------------------------------------------------------+
    |
    | (Repeat 4 times - 4 Transformer Decoder Layers)
    |
    v
+-------------------------------------------------------------+
|  TRANSFORMER DECODER LAYER                                  |
|                                                             |
|  +-------------------------------------------------------+  |
|  |  MULTI-HEAD SELF-ATTENTION (8 heads)                  |  |
|  |  +---------+ +---------+ +---------+ +---------+      |  |
|  |  | Head 1  | | Head 2  | | Head 3  | |  ...    |      |  |
|  |  | Q K V   | | Q K V   | | Q K V   | | Head 8  |      |  |
|  |  +---------+ +---------+ +---------+ +---------+      |  |
|  |  With Causal Mask (prevents attending to future)      |  |
|  +-------------------------------------------------------+  |
|                         |                                   |
|                         v                                   |
|  +-------------------------------------------------------+  |
|  |  ADD & LAYER NORM                                     |  |
|  |  LayerNorm(x + MultiHeadAttention(x))                |  |
|  +-------------------------------------------------------+  |
|                         |                                   |
|                         v                                   |
|  +-------------------------------------------------------+  |
|  |  FEED-FORWARD NETWORK                                 |  |
|  |  FFN(x) = ReLU(xW1 + b1)W2 + b2                      |  |
|  |  Dimensions: 256 -> 1024 -> 256                        |  |
|  +-------------------------------------------------------+  |
|                         |                                   |
|                         v                                   |
|  +-------------------------------------------------------+  |
|  |  ADD & LAYER NORM                                     |  |
|  |  LayerNorm(x + FFN(x))                               |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
    |
    v
+-------------------------------------------------------------+
|  LINEAR PROJECTION TO VOCABULARY                            |
|  Input:  (batch_size, seq_len, d_model=256)               |
|  Output: (batch_size, seq_len, vocab_size=10000)          |
+-------------------------------------------------------------+
    |
    v
LOGITS (predictions for next token)
```

#### Multi-Head Attention Detail

```
        INPUT: (batch, seq_len, d_model=256)
               |
    +----------+----------+----------+----------+
    |          |          |          |
+--------+ +--------+ +--------+ +--------+
|Linear Q| |Linear K| |Linear V| |  ...   |
+--------+ +--------+ +--------+ +--------+
    |          |          |
    v          v          v
+----------------------------------------------------+
|  Reshape to (batch, num_heads=8, seq_len, d_k=32) |
+----------------------------------------------------+
    |
    v
+----------------------------------------------------+
|  SCALED DOT-PRODUCT ATTENTION                      |
|                                                    |
|  Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V          |
|                                                    |
|  1. Compute scores: QK^T                          |
|  2. Scale by sqrt(d_k) (sqrt(32) = 5.66)                    |
|  3. Apply causal mask (set future to -inf)          |
|  4. Softmax to get attention weights              |
|  5. Multiply by V to get output                   |
+----------------------------------------------------+
    |
    v
+----------------------------------------------------+
|  Concatenate all heads                             |
|  Reshape back to (batch, seq_len, d_model)        |
+----------------------------------------------------+
    |
    v
+----------------------------------------------------+
|  Linear projection (output)                        |
+----------------------------------------------------+
    |
    v
  OUTPUT
```

#### Causal Mask

The causal mask ensures the model can only attend to previous positions:

```
For sequence length 5:

Attention Mask (1 = can attend, 0 = cannot attend):
       t1  t2  t3  t4  t5
    +--------------------+
t1  | 1   0   0   0   0 |  Token 1 can only see itself
t2  | 1   1   0   0   0 |  Token 2 can see t1, t2
t3  | 1   1   1   0   0 |  Token 3 can see t1, t2, t3
t4  | 1   1   1   1   0 |  Token 4 can see t1-t4
t5  | 1   1   1   1   1 |  Token 5 can see all previous
    +--------------------+

This prevents the model from "cheating" by seeing future tokens.
```

#### Transformer Model Parameters
- **Model dimension (d_model)**: 256
- **Number of heads**: 8
- **Head dimension (d_k)**: 32 (256/8)
- **Number of layers**: 4
- **Feed-forward dimension**: 1024
- **Dropout**: 0.1
- **Total parameters**: ~8-12M

**Key Characteristics:**
- Parallel processing (all positions at once)
- Self-attention captures long-range dependencies
- Positional encoding for sequence order
- Better at modeling complex dependencies
- More computationally intensive
- Generally achieves better perplexity

---

## Project Structure

```
.
├── data/
│   ├── raw/                    # Your input data files
│   │   └── polish_news.txt     # Example: domain-specific data
│   ├── processed/              # Generated after preprocessing
│   │   ├── tokenizer.json      # Trained BPE tokenizer
│   │   ├── tokenizer_metadata.json
│   │   ├── train_ids.pt        # Tokenized training data
│   │   ├── val_ids.pt          # Tokenized validation data
│   │   ├── test_ids.pt         # Tokenized test data
│   │   ├── metadata.json       # Dataset statistics
│   │   └── wikipedia_ids.pt    # (Optional) Wikipedia data
│   └── wikipedia/              # Out-of-domain evaluation data
│       └── polish_wiki.txt     # Polish Wikipedia sample
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── rnn_model.py           # LSTM language model
│   └── transformer_model.py   # Transformer language model
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   ├── tokenizer.py           # BPE tokenizer
│   ├── dataset.py             # PyTorch datasets and dataloaders
│   └── metrics.py             # Perplexity and evaluation metrics
│
├── scripts/                    # Executable scripts
│   ├── __init__.py
│   ├── preprocess_data.py     # Data preprocessing pipeline
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── generate.py            # Text generation
│
├── checkpoints/                # Saved model checkpoints
│   ├── rnn_best.pt            # Best RNN model
│   ├── rnn_epoch_N.pt         # RNN checkpoints per epoch
│   ├── transformer_best.pt    # Best Transformer model
│   └── transformer_epoch_N.pt # Transformer checkpoints
│
├── results/                    # Evaluation results
│   ├── rnn_metrics.json       # RNN training history
│   ├── rnn_eval_test.json     # RNN test evaluation
│   ├── rnn_eval_wikipedia.json # RNN Wikipedia evaluation
│   ├── rnn_generations.txt    # RNN generated text
│   ├── rnn_generations.json   # RNN generations (JSON)
│   └── transformer_*.json/txt # Same for Transformer
│
├── pyproject.toml             # Project dependencies (uv)
├── README.md                  # Lab assignment description
├── DOCS.md                    # This file
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.12+
- uv package manager
- Mac with Apple Silicon (or any system with CUDA/CPU)

### Setup

1. **Clone/navigate to the project directory**:
```bash
cd "/path/to/LAB1 13.10.2025"
```

2. **Install dependencies**:
```bash
uv sync
```

This installs:
- PyTorch (with MPS support for Mac)
- tokenizers (Hugging Face BPE tokenizer)
- numpy, matplotlib, tqdm
- datasets (for data handling)

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Complete Workflow

### Quick Reference

```bash
# 1. Preprocess data
python scripts/preprocess_data.py --input data/raw/polish_news.txt --config rnn --file-type txt

# 2. Train both models
python scripts/train.py --model rnn
python scripts/train.py --model transformer

# 3. Evaluate on test set (in-domain)
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data test

# 4. Evaluate on Wikipedia (out-of-domain) - optional
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data wikipedia
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data wikipedia

# 5. Generate text
python scripts/generate.py --model rnn --checkpoint checkpoints/rnn_best.pt \
  --prompts "Warszawa jest" "W Polsce" "Dzisiaj"
python scripts/generate.py --model transformer --checkpoint checkpoints/transformer_best.pt \
  --prompts "Warszawa jest" "W Polsce" "Dzisiaj"
```

---

## Detailed Usage Guide

### Step 1: Prepare Your Data

Your data should be Polish text from a specific domain (e.g., news, literature, social media).

**Data Format Options:**

1. **Plain text file** (`.txt`): One document per line
   ```
   To jest pierwszy dokument.
   To jest drugi dokument o polityce.
   Trzeci dokument o sporcie.
   ```

2. **JSONL file** (`.jsonl`): One JSON object per line
   ```json
   {"text": "To jest pierwszy dokument.", "metadata": "..."}
   {"text": "To jest drugi dokument o polityce.", "metadata": "..."}
   ```

**Where to get data:**
- [Speakleash](https://github.com/speakleash/speakleash) - Polish language corpus
- [OSCAR](https://oscar-corpus.com/) - Multilingual corpus
- [Polish Wikipedia dumps](https://dumps.wikimedia.org/plwiki/)
- Your own domain-specific corpus

**Place your data:**
```bash
# Create directory if needed
mkdir -p data/raw

# Copy your data
cp /path/to/your/data.txt data/raw/polish_news.txt
```

---

### Step 2: Preprocess Data

This step:
1. Loads your raw text
2. Splits into train (85%), validation (10%), test (5%)
3. Trains a BPE tokenizer on the training set
4. Tokenizes all splits
5. Saves processed data

**For text files:**
```bash
python scripts/preprocess_data.py \
  --input data/raw/polish_news.txt \
  --config rnn \
  --file-type txt
```

**For JSONL files:**
```bash
python scripts/preprocess_data.py \
  --input data/raw/data.jsonl \
  --config rnn \
  --file-type jsonl \
  --text-field text
```

**Arguments:**
- `--input`: Path to your data file
- `--config`: Model config type (`rnn` or `transformer`) - affects vocab size, seq length
- `--file-type`: `txt` or `jsonl`
- `--text-field`: For JSONL, the field containing text (default: `text`)

**Output:**
```
data/processed/
  ├── tokenizer.json              # Trained BPE tokenizer
  ├── tokenizer_metadata.json     # Tokenizer config
  ├── train_ids.pt                # Training sequences (List[List[int]])
  ├── val_ids.pt                  # Validation sequences
  ├── test_ids.pt                 # Test sequences
  └── metadata.json               # Dataset statistics
```

**Example output:**
```
================================================================================
DATA PREPROCESSING
================================================================================

Configuration: RNN(LSTM, layers=2, emb=256, hidden=512)
Device: mps

Loading data from data/raw/polish_news.txt...
Loaded 15000 documents
Sample text: W Warszawie odbyla sie konferencja prasowa...

Splitting data (train=0.85, val=0.1, test=0.05)...
Train: 12750 | Val: 1500 | Test: 750

Training tokenizer (vocab_size=10000)...
Tokenizer trained! Vocabulary size: 10000

Tokenizing data...
  Tokenizing training set... 100%|==========| 12750/12750
  Tokenizing validation set... 100%|==========| 1500/1500
  Tokenizing test set... 100%|==========| 750/750

Tokenization statistics:
  Train avg length: 87.3 tokens
  Val avg length: 86.9 tokens
  Test avg length: 88.1 tokens

Saving preprocessed data...
Preprocessing complete!
================================================================================
```

---

### Step 3: Train Models

Train both RNN and Transformer models on your preprocessed data.

**Train RNN:**
```bash
python scripts/train.py --model rnn
```

**Train Transformer:**
```bash
python scripts/train.py --model transformer
```

**Arguments:**
- `--model`: Model type (`rnn` or `transformer`)
- `--resume`: (Optional) Path to checkpoint to resume from

**Training process:**

1. Loads preprocessed data
2. Creates model with configured architecture
3. Trains for N epochs (default: 10)
4. Validates after each epoch
5. Saves checkpoints periodically and when validation improves
6. Tracks training time per epoch

**Example output:**
```
================================================================================
TRAINING RNN LANGUAGE MODEL
================================================================================

Configuration: RNN(LSTM, layers=2, emb=256, hidden=512)
Device: mps

Loading preprocessed data...
Train: 12750 | Val: 1500 | Test: 750

Creating dataloaders...

Initializing model...
Model parameters: 7,542,000

================================================================================
STARTING TRAINING
================================================================================

Epoch 1/10
--------------------------------------------------------------------------------
Training: 100%|==========| 399/399 [02:13<00:00, 2.98it/s, loss=6.2341]
Validating...

Epoch 1 Summary:
  Train Loss: 6.2341 | Train PPL: 509.87
  Val Loss:   6.0123 | Val PPL:   410.23
  LR: 0.001000 | Time: 153.2s
  Best model saved: checkpoints/rnn_best.pt

Epoch 2/10
--------------------------------------------------------------------------------
Training: 100%|==========| 399/399 [02:11<00:00, 3.03it/s, loss=5.8234]
...

================================================================================
TRAINING COMPLETE
================================================================================
Total training time: 1532.4s (25.5m)
Best validation loss: 4.8234
Best validation perplexity: 124.56

Metrics saved to: results/rnn_metrics.json
================================================================================
```

**During training:**
- Progress bars show current batch loss
- Validation runs after each epoch
- Learning rate adjusts based on validation loss (ReduceLROnPlateau)
- Checkpoints saved every 2 epochs (configurable)
- Best model (lowest val loss) always saved

**Checkpoints saved:**
```
checkpoints/
  ├── rnn_best.pt              # Best model (lowest val loss)
  ├── rnn_epoch_2.pt           # Checkpoint at epoch 2
  ├── rnn_epoch_4.pt           # Checkpoint at epoch 4
  └── ...
```

**Each checkpoint contains:**
- Model state dict (weights)
- Optimizer state dict
- Epoch number
- Training and validation loss
- Configuration

**Resume training:**
```bash
# If training was interrupted
python scripts/train.py --model rnn --resume checkpoints/rnn_epoch_4.pt
```

**Metrics saved:**
```json
{
  "train_losses": [6.2341, 5.8234, ...],
  "val_losses": [6.0123, 5.7456, ...],
  "train_perplexities": [509.87, 338.12, ...],
  "val_perplexities": [410.23, 312.45, ...],
  "learning_rates": [0.001, 0.001, 0.0005, ...],
  "epoch_times": [153.2, 151.8, ...],
  "total_time": 1532.4,
  "avg_epoch_time": 153.2,
  "best_val_loss": 4.8234,
  "best_val_perplexity": 124.56
}
```

---

### Step 4: Evaluate Models

Evaluate trained models on test data or Wikipedia.

**Evaluate on test set (in-domain):**
```bash
python scripts/evaluate.py \
  --model rnn \
  --checkpoint checkpoints/rnn_best.pt \
  --data test
```

**Arguments:**
- `--model`: Model type (`rnn` or `transformer`)
- `--checkpoint`: Path to model checkpoint
- `--data`: Dataset to evaluate on (`test`, `val`, or `wikipedia`)

**Example output:**
```
================================================================================
EVALUATING RNN LANGUAGE MODEL
================================================================================

Configuration: RNN(LSTM, layers=2, emb=256, hidden=512)
Device: mps

Loading test data...
Loaded 750 sequences

Loading checkpoint from checkpoints/rnn_best.pt...
Initializing model...
Model parameters: 7,542,000

Evaluating on test set...
100%|==============================| 24/24 [00:03<00:00, 7.23it/s]

================================================================================
EVALUATION RESULTS
================================================================================
Model: RNN
Checkpoint: checkpoints/rnn_best.pt
Data: test
Sequences: 750
Total tokens: 66,075
--------------------------------------------------------------------------------
Loss: 4.8567
Perplexity: 128.92
Evaluation time: 3.32s
Throughput: 19,903 tokens/s
================================================================================

Results saved to: results/rnn_eval_test.json
```

**Wikipedia Evaluation (Out-of-Domain):**

First, prepare Wikipedia data:

1. Download Polish Wikipedia sample to `data/wikipedia/polish_wiki.txt`

2. Tokenize it using your trained tokenizer:
```bash
cat > scripts/tokenize_wikipedia.py << 'EOF'
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.tokenizer import PolishTokenizer
from utils.dataset import load_text_file
from tqdm import tqdm

# Load trained tokenizer
tokenizer = PolishTokenizer()
tokenizer.load(Path('data/processed/tokenizer.json'))

# Load Wikipedia text
texts = load_text_file(Path('data/wikipedia/polish_wiki.txt'))

# Tokenize
wiki_ids = []
for text in tqdm(texts, desc="Tokenizing Wikipedia"):
    ids = tokenizer.encode(text, add_special_tokens=True)
    wiki_ids.append(ids)

# Save
torch.save(wiki_ids, Path('data/processed/wikipedia_ids.pt'))
print(f"Saved {len(wiki_ids)} Wikipedia sequences")
EOF

python scripts/tokenize_wikipedia.py
```

3. Evaluate models on Wikipedia:
```bash
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data wikipedia
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data wikipedia
```

**Expected behavior:**
- **In-domain (test)**: Lower perplexity (model has seen similar data)
- **Out-of-domain (Wikipedia)**: Higher perplexity (different domain)
- The difference shows how well the model generalizes

---

### Step 5: Generate Text

Generate text completions using trained models.

**Basic usage:**
```bash
python scripts/generate.py \
  --model rnn \
  --checkpoint checkpoints/rnn_best.pt \
  --prompts "Warszawa jest" "W Polsce" "Dzisiaj pogoda"
```

**From a file:**
```bash
# Create prompts file (one per line)
cat > data/prompts.txt << EOF
Warszawa jest
W Polsce mieszka
Dzisiaj pogoda
Sztuczna inteligencja
Uczenie maszynowe
Najnowsze wiadomosci
Prezydent Polski
W tym roku
Polska gospodarka
Historia Polski
EOF

python scripts/generate.py \
  --model rnn \
  --checkpoint checkpoints/rnn_best.pt \
  --prompts-file data/prompts.txt \
  --max-length 100 \
  --temperature 0.8 \
  --top-k 50
```

**Arguments:**
- `--model`: Model type (`rnn` or `transformer`)
- `--checkpoint`: Path to checkpoint
- `--prompts`: List of prompts (space-separated)
- `--prompts-file`: File with prompts (one per line)
- `--max-length`: Max tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
  - Lower (0.5-0.8): More focused, deterministic
  - Higher (1.0-1.5): More random, creative
- `--top-k`: Sample from top-k tokens only (default: 50)

**Example output:**
```
================================================================================
TEXT GENERATION - RNN MODEL
================================================================================

Device: mps

Loading tokenizer...
Loading checkpoint from checkpoints/rnn_best.pt...
Initializing model...
Model parameters: 7,542,000

================================================================================
GENERATING TEXT
================================================================================

Prompt 1/3:
  Input: "Warszawa jest"
  Output: "Warszawa jest stolica Polski i najwiekszym miastem kraju.
          W Warszawie znajduje sie wiele zabytkow i muzeow..."
  Time: 0.234s

Prompt 2/3:
  Input: "W Polsce"
  Output: "W Polsce mieszka okolo 38 milionow ludzi. Polska jest
          czlonkiem Unii Europejskiej od 2004 roku..."
  Time: 0.241s

Prompt 3/3:
  Input: "Dzisiaj pogoda"
  Output: "Dzisiaj pogoda jest sloneczna i ciepla. Temperatura
          osiagnie 25 stopni Celsjusza..."
  Time: 0.228s

================================================================================
SAVING RESULTS
================================================================================
Results saved to: results/rnn_generations.txt
JSON results saved to: results/rnn_generations.json
================================================================================
```

**Output files:**

`results/rnn_generations.txt` (human-readable):
```
Text Generation Results - RNN Model
================================================================================
Checkpoint: checkpoints/rnn_best.pt
Max length: 100
Temperature: 0.8
Top-k: 50
================================================================================

PROMPT 1:
Warszawa jest

COMPLETION:
Warszawa jest stolica Polski i najwiekszym miastem kraju...

Time: 0.234s
--------------------------------------------------------------------------------
...
```

`results/rnn_generations.json` (programmatic):
```json
[
  {
    "prompt": "Warszawa jest",
    "completion": "stolica Polski i najwiekszym miastem kraju...",
    "full_text": "Warszawa jest stolica Polski...",
    "time": 0.234
  },
  ...
]
```

---

### Step 6: Compare Models

After training and evaluating both models:

```bash
# Evaluate both on test
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data test

# Evaluate both on Wikipedia
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data wikipedia
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data wikipedia

# Generate from both
python scripts/generate.py --model rnn --checkpoint checkpoints/rnn_best.pt --prompts-file data/prompts.txt
python scripts/generate.py --model transformer --checkpoint checkpoints/transformer_best.pt --prompts-file data/prompts.txt
```

**Create comparison table for your report:**

| Metric | RNN | Transformer |
|--------|-----|-------------|
| **Architecture** | 2-layer LSTM | 4-layer Transformer |
| **Parameters** | ~7.5M | ~10.2M |
| **Test Perplexity** | 128.92 | 98.45 |
| **Wikipedia Perplexity** | 187.34 | 156.78 |
| **Training Time** | 25.5 min | 42.3 min |
| **Inference Speed** | 19,903 tokens/s | 15,234 tokens/s |
| **Generation Quality** | Good for short text | Better coherence |

---

## Configuration

All hyperparameters are in `utils/config.py`.

### RNN Configuration

```python
@dataclass
class RNNConfig(Config):
    # Model architecture
    embedding_dim: int = 256        # Token embedding dimension
    hidden_dim: int = 512           # LSTM hidden state dimension
    num_layers: int = 2             # Number of LSTM layers
    dropout: float = 0.3            # Dropout probability
    rnn_type: str = "LSTM"          # "LSTM" or "GRU"

    # ~5-8M parameters
```

### Transformer Configuration

```python
@dataclass
class TransformerConfig(Config):
    # Model architecture
    embedding_dim: int = 256        # Token embedding dimension (d_model)
    num_heads: int = 8              # Number of attention heads
    num_layers: int = 4             # Number of transformer layers
    ff_dim: int = 1024              # Feed-forward hidden dimension
    dropout: float = 0.1            # Dropout probability

    # ~8-12M parameters
```

### General Configuration

```python
@dataclass
class Config:
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    # Data
    vocab_size: int = 10000         # Vocabulary size
    max_seq_length: int = 128       # Maximum sequence length
    train_split: float = 0.85       # Train/val/test split
    val_split: float = 0.10
    test_split: float = 0.05

    # Training
    batch_size: int = 32            # Batch size for training
    num_epochs: int = 10            # Number of training epochs
    learning_rate: float = 0.001    # Initial learning rate
    weight_decay: float = 0.01      # Weight decay (L2 regularization)
    gradient_clip: float = 1.0      # Gradient clipping threshold

    # Evaluation
    eval_batch_size: int = 64       # Batch size for evaluation

    # Generation
    max_gen_length: int = 100       # Max tokens to generate
    temperature: float = 1.0        # Sampling temperature
    top_k: int = 50                 # Top-k sampling
```

**To modify configuration:**

Edit `utils/config.py` directly:

```python
# For smaller models (faster training on Mac)
embedding_dim: int = 128
hidden_dim: int = 256
num_layers: int = 1
batch_size: int = 16

# For larger models (better performance, slower)
embedding_dim: int = 512
hidden_dim: int = 1024
num_layers: int = 3
batch_size: int = 64
```

---

## Evaluation Metrics

### Perplexity

Perplexity measures how well the model predicts the next token. Lower is better.

**Formula:**
```
Perplexity = exp(CrossEntropyLoss)
```

**Interpretation:**
- **Perplexity = 1**: Perfect prediction (impossible in practice)
- **Perplexity = 100**: On average, the model is as confused as if choosing uniformly from 100 options
- **Perplexity = 10,000**: Very poor (similar to random guessing from full vocab)

**Typical ranges:**
- **State-of-the-art large models** (GPT-3, etc.): 10-30
- **Good small models**: 50-150
- **Baseline models**: 200-500
- **Random baseline**: ~vocab_size (10,000 in our case)

**In-domain vs Out-of-domain:**
- In-domain: Lower perplexity (model has seen similar text)
- Out-of-domain: Higher perplexity (different vocabulary, style, topics)

### Training Time

Total time spent training the model (all epochs).

### Inference Time

Time to process test data, measured as:
- **Total time**: Seconds to evaluate all test data
- **Throughput**: Tokens per second

---

## Tips & Troubleshooting

### Mac-Specific Tips

1. **Check MPS availability:**
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

2. **Monitor memory:**
   - Use Activity Monitor to watch memory usage
   - If running out of memory, reduce `batch_size`

3. **Optimization:**
   - MPS is faster than CPU but slower than CUDA
   - Training might take 10-30 minutes per model
   - Consider smaller models for experimentation

### Common Issues

**Out of Memory:**
```
RuntimeError: MPS backend out of memory
```
**Solutions:**
- Reduce `batch_size` (try 16 or 8)
- Reduce `max_seq_length` (try 64)
- Use smaller model (reduce layers, hidden dims)

**Slow Training:**
**Solutions:**
- Use smaller dataset (subsample your data)
- Reduce `num_epochs`
- Use smaller model
- Reduce `max_seq_length`

**Poor Perplexity:**
Perplexity > 500 or not improving
**Solutions:**
- Train longer (more epochs)
- Increase model size
- Use more/better training data
- Check data quality (tokenization issues?)
- Adjust learning rate

**Bad Text Generation:**
Gibberish or repetitive output
**Solutions:**
- Train longer (model not converged)
- Adjust `temperature` (try 0.7-0.9)
- Adjust `top_k` (try 30-100)
- Check if model trained properly (low perplexity?)

**Tokenizer Issues:**
```
ValueError: Tokenizer not trained yet!
```
**Solutions:**
- Run preprocessing first
- Check `data/processed/tokenizer.json` exists

---

## Expected Results

### For Small Polish Dataset (~10-15k documents)

#### In-Domain (Test Set)

| Model | Perplexity | Training Time | Inference Speed |
|-------|-----------|---------------|-----------------|
| RNN (LSTM) | 80-150 | 10-20 min | 15,000-25,000 tokens/s |
| Transformer | 60-120 | 15-30 min | 10,000-20,000 tokens/s |

**Notes:**
- Transformer usually achieves 15-25% better perplexity
- RNN trains faster (sequential is simpler)
- RNN may be faster at inference on CPU/MPS

#### Out-of-Domain (Wikipedia)

| Model | Perplexity | Perplexity Increase |
|-------|-----------|---------------------|
| RNN | 150-250 | +70-100 |
| Transformer | 120-200 | +60-80 |

**Notes:**
- Both models show increased perplexity on out-of-domain data
- This is expected and normal
- Smaller increase = better generalization

#### Generation Quality

**RNN:**
- Good for short completions
- May lose coherence after 50-100 tokens
- Faster generation
- Sometimes repetitive

**Transformer:**
- Better long-range coherence
- More diverse vocabulary usage
- Better grammar and style
- May be slower

### Sample Output Quality

**Prompt:** "Warszawa jest"

**RNN output:**
```
Warszawa jest stolica Polski. Miasto ma okolo 2 miliony mieszkancow.
W Warszawie znajduja sie liczne zabytki i muzea. Stolica Polski jest
waznym centrum kulturalnym i ekonomicznym.
```

**Transformer output:**
```
Warszawa jest najwiekszym miastem Polski i jednoczesnie jej stolica.
Polozona na Mazowszu, nad Wisla, stanowi centrum polityczne, gospodarcze
i kulturalne kraju. W 2024 roku liczba mieszkancow przekroczyla 1,8 miliona.
```

**Observation:** Transformer typically produces more coherent, detailed, and contextually appropriate text.

---

## For Your Report

### Required Sections

1. **Model Architecture**
   - Describe RNN and Transformer architectures
   - Include parameter counts
   - Reference the diagrams in this documentation

2. **Dataset Description**
   - Domain and source
   - Number of documents
   - Train/val/test split sizes
   - From `data/processed/metadata.json`

3. **Training Details**
   - Hyperparameters used
   - Training time per model
   - From `results/*_metrics.json`

4. **Evaluation Results**
   - Perplexity on test set (in-domain)
   - Perplexity on Wikipedia (out-of-domain)
   - Comparison table
   - From `results/*_eval_*.json`

5. **Time Comparison**
   - Training time
   - Inference time and throughput
   - From metrics and evaluation files

6. **Generated Examples**
   - At least 10 prompts with completions
   - Show outputs from both models
   - Discuss quality differences
   - From `results/*_generations.txt`

7. **Discussion**
   - Which model performed better and why?
   - How did models handle out-of-domain data?
   - Challenges encountered
   - Insights learned

---

## Quick Commands Reference

```bash
# Setup
uv sync

# Preprocess
python scripts/preprocess_data.py --input data/raw/file.txt --config rnn --file-type txt

# Train
python scripts/train.py --model rnn
python scripts/train.py --model transformer

# Evaluate
python scripts/evaluate.py --model rnn --checkpoint checkpoints/rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer_best.pt --data test

# Generate
python scripts/generate.py --model rnn --checkpoint checkpoints/rnn_best.pt --prompts "Your prompt"

# Resume training
python scripts/train.py --model rnn --resume checkpoints/rnn_epoch_4.pt

# Custom generation
python scripts/generate.py --model transformer --checkpoint checkpoints/transformer_best.pt \
  --prompts-file data/prompts.txt --max-length 150 --temperature 0.7 --top-k 40
```

---

## References

- **Attention Is All You Need** (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997)
- **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)

---

For questions or issues, check the troubleshooting section or review the code in `models/`, `utils/`, and `scripts/`.
