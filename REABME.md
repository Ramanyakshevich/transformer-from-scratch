# Custom Transformer Encoder Architecture from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bFKb6pwJxs7UewD8Zh6X4Ipt8KRKZZuL)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![HuggingFace](https://img.shields.io/badge/Transformers-BERT%20Tokenizer-yellow)

A modular implementation of a **Transformer Encoder architecture built completely from scratch** using pure PyTorch (`torch.nn`), applied to binary sentiment classification on 60,000 balanced Amazon Movie Reviews.

---

## Key Architectural Highlights

* **Scaled Dot-Product Multi-Head Attention:** Vectorized implementation of $QK^T / \sqrt{d_k}$ with attention masking (`-inf` prior to softmax) to eliminate variable-length padding noise.
* **Sinusoidal Positional Encoding:** Analytical sinusoidal and cosinusoidal coordinate encodings.
* **Residual Connections & LayerNorm:** Pre/post-attention additions and feed-forward MLP projections with dropout regularization.
* **Masked Mean-Pooling Head:** Aggregation of token representations with zero-division protection (`torch.clamp`) for robust sequence-level classification.
* **Training Stability:** Gradient norm clipping (`clip_grad_norm_`) paired with `AdamW` weight decay.

---

## Dataset Pipeline

* **Source:** `rohan2810/amazon-movies-meta-reviews-merged` via Hugging Face.
* **Balancing:** 30,000 positive and 30,000 negative reviews (neutral ratings removed).
* **Tokenization:** Subword tokenization via `bert-base-uncased` with truncation and padding to 128 tokens.

---

## Running in the Cloud

Since training transformer architectures requires GPU acceleration, you can run the interactive end-to-end training notebook with one click:

👉 **[Open Interactive Notebook in Google Colab](https://colab.research.google.com/drive/1bFKb6pwJxs7UewD8Zh6X4Ipt8KRKZZuL)**