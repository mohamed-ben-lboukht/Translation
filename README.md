# Transformer Translation Machine: English to Turkish

A from-scratch implementation of a transformer-based neural machine translation (NMT) model following the seminal paper **"Attention Is All You Need"** by Vaswani et al. This project demonstrates a complete English-to-Turkish translation pipeline using the transformer architecture.

---

## 🚀 Project Overview

This implementation builds every core transformer component manually to illustrate a deep understanding of the architecture:

- Multi-head self-attention mechanisms  
- Positional encoding  
- Encoder-decoder architecture  
- Feed-forward networks  
- Layer normalization  
- Custom learning rate scheduling  

The model is trained on English–Turkish sentence pairs and provides functional translation performance.

---

## 🧱 Architecture Details

### Model Configuration

| Parameter             | Value         |
|-----------------------|---------------|
| Model Dimension       | 128           |
| Layers (Encoder/Decoder) | 4         |
| Attention Heads       | 4             |
| Head Dimension        | 32            |
| Vocabulary Size       | 16,000 tokens |
| Max Sequence Length   | 512 tokens    |
| Dropout Rate          | 0.1           |

---

## 🔑 Key Components

### Encoder
- Multi-layer encoder with self-attention
- Positional embeddings
- Residual connections + LayerNorm
- Feed-forward networks with ReLU

### Decoder
- Masked multi-head self-attention
- Cross-attention to encoder output
- Causal masking for autoregressive decoding
- Final projection layer to vocab size

### Attention Mechanism
- Scaled dot-product attention
- Multi-head attention with projection matrices
- Proper scaling using √dₖ for stability

---

## 🛠️ Features Implemented

### Transformer Components
- Multi-Head Attention
- Positional Encoding (learned embeddings)
- Layer Normalization (pre-norm)
- Residual Connections
- Feed-Forward Networks

### Training Features
- Noam Learning Rate Scheduler
- Gradient Clipping
- Label Smoothing (via `CrossEntropyLoss`)
- `torch.compile()` for optimization

### Tokenization
- SentencePiece (BPE) for subword units
- Special Tokens: BOS / EOS
- Padding for batch training

### Inference
- Autoregressive decoding
- Top-k sampling for diversity
- Temperature control for randomness

---

## 📦 Requirements

### Dependencies
- `PyTorch` (with CUDA support)
- `SentencePiece`
- `NumPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`
- `tqdm`
- `scikit-learn`

### Hardware
- CUDA-compatible GPU (recommended)
- Adequate RAM and disk space for model checkpoints and dataset

---

## 📚 Dataset

- Cleaned English-Turkish sentence pairs
- Preprocessing:
  - SentencePiece BPE tokenization
  - Padding for batching
  - BOS/EOS token addition
  - Train/validation split (80/20)

---

## ⚙️ Training

### Hyperparameters

| Parameter         | Value       |
|-------------------|-------------|
| Batch Size        | 32          |
| Learning Rate     | 1.0         |
| Warmup Steps      | 4,000       |
| Epochs            | 5           |
| Optimizer         | AdamW (β₁=0.9, β₂=0.98) |

### Features
- Noam Scheduler for stable learning
- Gradient Clipping
- Cross-Entropy Loss with `ignore_index` for padding
- Training progress tracking with loss/learning rate plots

---

## 📈 Model Performance

- Functional English → Turkish translations
- Captures Turkish morphology effectively
- Learns contextual alignment in typical phrases

### Sample Translations
```text
Input:  "Hello, how are you?"       → Output: "Merhaba, nasılsın?"
Input:  "Thanks."                   → Output: "Teşekkürler."
Input:  "Let's try something."      → Output: "Bir şey yapalım."
```
---
### References
- Vaswani, A., et al. "Attention Is All You Need." Advances in Neural Information Processing Systems 30 (2017).
- The implementation follows the original transformer paper architecture
- SentencePiece tokenization for subword handling
- PyTorch framework for deep learning implementation

### Acknowledgments
This implementation serves as an educational project to understand the transformer architecture from first principles. The implementation demonstrates how to build a complete neural machine translation system following the groundbreaking "Attention Is All You Need" paper.
