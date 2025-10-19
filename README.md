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
7. [Training Visualization & Plotting](#training-visualization--plotting)
8. [Configuration](#configuration)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Tips & Troubleshooting](#tips--troubleshooting)
11. [Expected Results](#expected-results)

---

## Project Overview

This project implements and compares two neural language model architectures for causal language modeling (next token prediction) on Polish text:

1. **RNN-based model** using LSTM (Long Short-Term Memory)
2. **Transformer-based model** using multi-head self-attention

### Key Features

- **Speakleash Integration** - Built-in dataset downloader for Polish language corpora
- **Causal Language Modeling** - Predicts the next token in a sequence
- **BPE Tokenization** - Custom Byte-Pair Encoding tokenizer trained on your data
- **Dual Evaluation** - In-domain (test set) and out-of-domain (Wikipedia)
- **Perplexity Metrics** - Standard metric for language model quality
- **Time Tracking** - Training and inference time measurement
- **Text Generation** - Autoregressive text generation with temperature and top-k sampling
- **Training Visualization** - Automated plotting of loss, perplexity, learning rate, and training time
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
│   ├── raw/                              # Your input data files
│   │   └── shopping_1_general_corpus.txt # Downloaded datasets from Speakleash
│   ├── processed/                        # Generated after preprocessing
│   │   ├── shopping_1_general_corpus_tokenizer.json     # Trained BPE tokenizer
│   │   ├── shopping_1_general_corpus_train_ids.pt       # Tokenized training data
│   │   ├── shopping_1_general_corpus_val_ids.pt         # Tokenized validation data
│   │   ├── shopping_1_general_corpus_test_ids.pt        # Tokenized test data
│   │   └── shopping_1_general_corpus_metadata.json      # Dataset statistics
│   └── wikipedia/                        # Out-of-domain evaluation data
│       └── polish_wiki.txt               # Polish Wikipedia sample
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
│   ├── metrics.py             # Perplexity and evaluation metrics
│   └── plotting.py            # Training visualization utilities
│
├── scripts/                    # Executable scripts
│   ├── __init__.py
│   ├── preprocess_data.py     # Data preprocessing pipeline
│   ├── train.py               # Model training (with automatic plotting)
│   ├── evaluate.py            # Model evaluation
│   ├── generate.py            # Text generation
│   └── visualize_metrics.py   # Standalone visualization and comparison
│
├── checkpoints/                                           # Saved model checkpoints
│   ├── shopping_1_general_corpus_rnn_best.pt             # Best RNN model
│   ├── shopping_1_general_corpus_rnn_epoch_N.pt          # RNN checkpoints per epoch
│   ├── shopping_1_general_corpus_transformer_best.pt     # Best Transformer model
│   └── shopping_1_general_corpus_transformer_epoch_N.pt  # Transformer checkpoints
│
├── results/                                                # Evaluation results
│   ├── plots/                                             # Training visualization plots
│   │   ├── shopping_1_general_corpus_rnn_loss_epoch_N.png
│   │   ├── shopping_1_general_corpus_rnn_perplexity_epoch_N.png
│   │   ├── shopping_1_general_corpus_rnn_combined_metrics_epoch_N.png
│   │   ├── shopping_1_general_corpus_transformer_*.png
│   │   └── model_comparison.png
│   ├── shopping_1_general_corpus_rnn_metrics.json        # RNN training history
│   ├── shopping_1_general_corpus_rnn_eval_test.json      # RNN test evaluation
│   ├── shopping_1_general_corpus_rnn_eval_wikipedia.json # RNN Wikipedia evaluation
│   ├── shopping_1_general_corpus_rnn_generations.txt     # RNN generated text
│   ├── shopping_1_general_corpus_rnn_generations.json    # RNN generations (JSON)
│   └── shopping_1_general_corpus_transformer_*.json/txt  # Same for Transformer
│
├── main.py                    # Dataset download script (Speakleash integration)
├── pyproject.toml             # Project dependencies (uv)
├── README.md                  # Complete documentation (this file)
└── .gitignore
```

**Note:** All processed data, checkpoints, and results are prefixed with the dataset name (e.g., `shopping_1_general_corpus_`). This allows you to work with multiple datasets simultaneously without file conflicts.

---

## Installation

### Prerequisites

- Python 3.12+
- uv package manager
- Mac with Apple Silicon (or any system with CUDA/CPU)

### Setup

1. **Clone/navigate to the project directory**:
```bash
cd "rnn-transformer-based-llm"
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
# 1. Download dataset from Speakleash
python main.py --list  # List available datasets
python main.py shopping_1_general_corpus  # Download dataset

# 2. Preprocess data (dataset name auto-extracted from filename)
python scripts/preprocess_data.py --input data/raw/shopping_1_general_corpus.txt --config rnn --file-type txt

# 3. Train both models (dataset auto-detected if only one exists)
python scripts/train.py --model rnn
python scripts/train.py --model transformer

# With multiple datasets, specify which one:
python scripts/train.py --model rnn --dataset shopping_1_general_corpus

# 4. Evaluate on test set (in-domain)
python scripts/evaluate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --data test

# 5. Evaluate on out-of-domain dataset (e.g., plwiki) - optional
python scripts/evaluate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --data out-of-domain --eval-dataset plwiki
python scripts/evaluate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --data out-of-domain --eval-dataset plwiki

# 6. Generate text
python scripts/generate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt \
  --prompts "Warszawa jest" "W Polsce" "Dzisiaj"
python scripts/generate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt \
  --prompts "Warszawa jest" "W Polsce" "Dzisiaj"
```

---

## Detailed Usage Guide

### Step 1: Download Dataset from Speakleash

This project uses the [Speakleash](https://github.com/speakleash/speakleash) library to download Polish language datasets. Speakleash provides curated Polish corpora from various domains (news, social media, literature, shopping, etc.).

**List available datasets:**
```bash
python main.py --list
```

This will show all available datasets with information about:
- Dataset name
- Category (Internet, Literature, etc.)
- Size in GB
- Number of documents and words
- License information

**Example output:**
```
Name: shopping_1_general_corpus
Category: Internet
Size: 2.22 GB
Documents: 2,105,419
Words: 1,551,865,826
License: conditional license
```

**Download a dataset:**
```bash
python main.py shopping_1_general_corpus
```

The script will:
1. Display dataset information and disclaimer
2. Ask for confirmation
3. Download the dataset (progress bar shown)
4. Save it to `data/raw/shopping_1_general_corpus.txt` (one document per line)
5. Show next steps for preprocessing

**Popular datasets for language modeling:**
- `shopping_1_general_corpus` - Product reviews and descriptions from ceneo.pl (~2.2GB, 2M docs)
- Other datasets available via `--list` command

---

### Step 2: Preprocess Data

This step:
1. Loads your raw text
2. Splits into train (85%), validation (10%), test (5%)
3. Trains a BPE tokenizer on the training set
4. Tokenizes all splits
5. Saves processed data

**Preprocess downloaded dataset:**
```bash
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --config rnn \
  --file-type txt
```

**Arguments:**
- `--input`: Path to downloaded dataset file from `data/raw/`
- `--config`: Model config type (`rnn` or `transformer`) - affects vocab size, seq length
- `--file-type`: Always `txt` for Speakleash datasets

**Output:**
```
data/processed/
  ├── shopping_1_general_corpus_tokenizer.json        # Trained BPE tokenizer
  ├── shopping_1_general_corpus_train_ids.pt          # Training sequences
  ├── shopping_1_general_corpus_val_ids.pt            # Validation sequences
  ├── shopping_1_general_corpus_test_ids.pt           # Test sequences
  └── shopping_1_general_corpus_metadata.json         # Dataset statistics
```

**Note:** All files are prefixed with the dataset name (extracted from the input filename). This allows you to preprocess and train on multiple datasets without file conflicts.

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
- `--dataset`: (Optional) Dataset name - auto-detected if only one preprocessed dataset exists
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
  ├── shopping_1_general_corpus_rnn_best.pt         # Best model (lowest val loss)
  ├── shopping_1_general_corpus_rnn_epoch_2.pt      # Checkpoint at epoch 2
  ├── shopping_1_general_corpus_rnn_epoch_4.pt      # Checkpoint at epoch 4
  └── ...
```

**Note:** Checkpoint files are prefixed with `{dataset}_{model}` to support training multiple models on different datasets.

**Each checkpoint contains:**
- Model state dict (weights)
- Optimizer state dict
- Epoch number
- Training and validation loss
- Configuration

**Resume training:**
```bash
# If training was interrupted
python scripts/train.py --model rnn --resume checkpoints/shopping_1_general_corpus_rnn_epoch_4.pt
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

Evaluate trained models on test data or other datasets (out-of-domain).

**Evaluate on test set (in-domain):**
```bash
python scripts/evaluate.py \
  --model rnn \
  --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt \
  --data test
```

**Arguments:**
- `--model`: Model type (`rnn` or `transformer`)
- `--checkpoint`: Path to model checkpoint
- `--data`: Dataset to evaluate on (`test`, `val`, or `out-of-domain`)
- `--dataset`: (Optional) Model's training dataset name - auto-detected if only one exists
- `--eval-dataset`: (Required for `out-of-domain`) Dataset name to evaluate on (e.g., `plwiki`)

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

**Out-of-Domain Evaluation:**

To evaluate a model on a different dataset (e.g., a model trained on shopping forums evaluated on Wikipedia):

1. **Download and preprocess the evaluation dataset** (e.g., Polish Wikipedia):
```bash
# Download dataset from Speakleash
python main.py plwiki

# Preprocess it with the SAME configuration as your training data
python scripts/preprocess_data.py \
  --input data/raw/plwiki.txt \
  --config rnn \
  --file-type txt
```

2. **Evaluate your trained model on the new dataset:**
```bash
# Model trained on shopping_1_general_corpus, evaluated on plwiki
python scripts/evaluate.py \
  --model rnn \
  --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt \
  --data out-of-domain \
  --eval-dataset plwiki

# Compare with transformer
python scripts/evaluate.py \
  --model transformer \
  --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt \
  --data out-of-domain \
  --eval-dataset plwiki
```

**Results will be saved as:**
- `results/shopping_1_general_corpus_rnn_eval_ood_plwiki.json`
- `results/shopping_1_general_corpus_transformer_eval_ood_plwiki.json`

**Expected behavior:**
- **In-domain (test)**: Lower perplexity (model has seen similar data)
- **Out-of-domain (different dataset)**: Higher perplexity (different domain/style)
- The difference shows how well the model generalizes across domains

---

### Step 5: Generate Text

Generate text completions using trained models.

**Basic usage:**
```bash
python scripts/generate.py \
  --model rnn \
  --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt \
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
  --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt \
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
- `--dataset`: (Optional) Dataset name - auto-detected if only one exists
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

## Training Visualization & Plotting

The project includes an automated plotting system that tracks and visualizes training metrics in real-time. Plots are automatically generated during training to help you monitor model performance.

### Features

**Automatic Plot Generation During Training**

When you run `python scripts/train.py --model rnn` or `python scripts/train.py --model transformer`, plots are automatically generated and saved after each epoch (by default).

**Generated Plots**

The system creates 6 different types of plots:

1. **Loss Plot** - Training and validation loss over epochs
2. **Perplexity Plot** - Training and validation perplexity over epochs
3. **Learning Rate Plot** - Learning rate schedule (log scale)
4. **Epoch Times Plot** - Time taken per epoch with average line
5. **Combined Metrics Plot** - All metrics in a 2x2 grid
6. **Overfitting Analysis** - Train vs val loss with generalization gap visualization
7. **Summary Statistics** - Text-based summary of key metrics (generated at the end)

**Plot Location**

All plots are saved in: `results/plots/`

- Individual plots: `results/plots/{model}_loss_epoch_{N}.png`
- Combined plots: `results/plots/{model}_combined_metrics_epoch_{N}.png`
- Final summary: `results/plots/{model}_summary_stats.png`

### Usage

**During Training (Automatic)**

Plots are automatically generated when you train:

```bash
python scripts/train.py --model rnn
# Plots automatically created in results/plots/ after each epoch
```

Example output:
```
Epoch 5 Summary:
  Train Loss: 4.2341 | Train PPL: 68.87
  Val Loss:   4.1123 | Val PPL:   61.23
  LR: 0.001000 | Time: 153.2s
  Best model saved: checkpoints/rnn_best.pt

Generating training plots (Epoch 5)...
Plots saved to: /path/to/results/plots
```

**Control Plot Frequency**

Edit `utils/config.py` to control how often plots are generated:

```python
plot_every_n_epochs: int = 1  # Generate plots every N epochs
```

- Set to `1`: Plot after every epoch (default, recommended)
- Set to `2`: Plot every 2 epochs (faster training, less I/O)
- Set to `5`: Plot every 5 epochs (minimal overhead)

**Manual Visualization from Saved Metrics**

Recreate plots from saved metrics JSON files:

```bash
# Visualize specific model
python scripts/visualize_metrics.py --metrics results/rnn_metrics.json

# Visualize with custom name
python scripts/visualize_metrics.py --metrics results/rnn_metrics.json --model-name "MyRNN"
```

**Compare RNN vs Transformer**

Create side-by-side comparison plots:

```bash
# Compare both models
python scripts/visualize_metrics.py --compare

# With custom paths
python scripts/visualize_metrics.py --compare \
  --rnn-metrics results/rnn_metrics.json \
  --transformer-metrics results/transformer_metrics.json
```

This creates:
- `results/plots/model_comparison.png` - Side-by-side loss and perplexity comparison
- Summary statistics table in terminal

### Understanding the Plots

**1. Loss Plot**
- **X-axis**: Epoch number
- **Y-axis**: Cross-entropy loss
- **Lines**: Blue (train), Red (validation)
- **Purpose**: Primary metric for training convergence
- **What to look for**: Both lines should decrease; if validation stops decreasing while training continues, you may be overfitting

**2. Perplexity Plot**
- **X-axis**: Epoch number
- **Y-axis**: Perplexity (exp of loss)
- **Lines**: Blue (train), Red (validation)
- **Purpose**: More interpretable metric (lower is better)
- **What to look for**: Lower values indicate better predictions; validation perplexity should track training perplexity

**3. Learning Rate Plot**
- **X-axis**: Epoch number
- **Y-axis**: Learning rate (log scale)
- **Purpose**: Verify learning rate scheduling
- **What to look for**: Should decrease in steps when validation loss plateaus (ReduceLROnPlateau)

**4. Epoch Times Plot**
- **X-axis**: Epoch number
- **Y-axis**: Time in seconds
- **Purpose**: Detect performance issues or slowdowns
- **What to look for**: Consistent times; sudden increases may indicate memory issues

**5. Combined Metrics Plot**
- **Layout**: 2x2 grid with all metrics
- **Purpose**: Single comprehensive view of training progress
- **Use this for**: Quick overview of how training is going

**6. Overfitting Analysis Plot**
- **Lines**: Train and validation loss
- **Shaded area**: Generalization gap (orange)
- **Purpose**: Detect if model is overfitting
- **What to look for**: If gap grows quickly, consider:
  - Increasing dropout
  - Reducing model size
  - Adding regularization
  - Getting more training data

**7. Summary Statistics**
- **Format**: Text-based summary
- **Contains**: Best metrics, total time, final metrics, improvement stats
- **Purpose**: Quick reference for report writing

### Example Workflow

```bash
# 1. Train RNN model (plots generated automatically)
python scripts/train.py --model rnn
# Check results/plots/ for rnn_combined_metrics_epoch_10.png

# 2. Train Transformer model (plots generated automatically)
python scripts/train.py --model transformer
# Check results/plots/ for transformer_combined_metrics_epoch_10.png

# 3. Compare both models
python scripts/visualize_metrics.py --compare
# Check results/plots/model_comparison.png

# 4. View plots
open results/plots/rnn_combined_metrics_epoch_10.png
open results/plots/transformer_combined_metrics_epoch_10.png
open results/plots/model_comparison.png
```

### Tips for Using Plots

1. **Monitor during training** - Open the plots folder and refresh to see latest updates

2. **Use combined metrics plot** - This gives you the best overview at a glance

3. **Watch for overfitting early** - Check the overfitting analysis plot regularly

4. **Compare final results** - Always run the comparison script after training both models

5. **Use plots in your report** - All plots are 150 DPI, suitable for academic papers

6. **Check learning rate changes** - Learning rate should decrease when validation loss plateaus

### Troubleshooting

**Issue**: No plots generated
- **Solution**: Check that `plot_every_n_epochs` in config is not too high

**Issue**: Plots folder doesn't exist
- **Solution**: The folder is created automatically, but ensure `results/` directory exists

**Issue**: Want to disable plotting temporarily
- **Solution**: Set `plot_every_n_epochs` to a very high number (e.g., 999) in `utils/config.py`

**Issue**: Plots look strange after resuming training
- **Solution**: This is normal - plots show all epochs including resumed ones

### Customization

To modify plot appearance, edit `utils/plotting.py`:

- **Colors**: Change line colors in plot functions (e.g., `'b-o'` for blue, `'r-s'` for red)
- **Figure size**: Modify `figsize` parameter (e.g., `figsize=(10, 6)`)
- **DPI**: Change `dpi=150` to higher/lower resolution
- **Font sizes**: Adjust `fontsize` parameters
- **Grid style**: Modify `grid()` parameters

---

### Step 6: Compare Models

After training and evaluating both models:

```bash
# Evaluate both on test
python scripts/evaluate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --data test

# Evaluate both on Wikipedia
python scripts/evaluate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --data wikipedia
python scripts/evaluate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --data wikipedia

# Generate from both
python scripts/generate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --prompts-file data/prompts.txt
python scripts/generate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --prompts-file data/prompts.txt
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

# Evaluate (dataset auto-detected)
python scripts/evaluate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --data test
python scripts/evaluate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt --data test

# Generate
python scripts/generate.py --model rnn --checkpoint checkpoints/shopping_1_general_corpus_rnn_best.pt --prompts "Your prompt"

# Resume training
python scripts/train.py --model rnn --resume checkpoints/shopping_1_general_corpus_rnn_epoch_4.pt

# Custom generation
python scripts/generate.py --model transformer --checkpoint checkpoints/shopping_1_general_corpus_transformer_best.pt \
  --prompts-file data/prompts.txt --max-length 150 --temperature 0.7 --top-k 40

# Visualize training metrics (plots generated automatically during training)
python scripts/visualize_metrics.py --metrics results/shopping_1_general_corpus_rnn_metrics.json
python scripts/visualize_metrics.py --compare  # Compare RNN vs Transformer
```

---

## References

- **Attention Is All You Need** (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997)
- **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)

---

For questions or issues, check the troubleshooting section or review the code in `models/`, `utils/`, and `scripts/`.
