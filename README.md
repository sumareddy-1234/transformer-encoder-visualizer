# 🧠 Transformer Encoder Visualizer

An end-to-end implementation of a Transformer Encoder from scratch using PyTorch and Streamlit for interpretability and attention visualization.

---

# 📌 Project Overview

This project demonstrates the internal mechanics of Transformer Encoders by implementing:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encodings
- Transformer Encoder Layers
- Attention Heatmap Visualization
- Attention Entropy Analysis
- Token Attribution

The project also includes an interactive Streamlit dashboard for exploring attention patterns across layers and heads.

---

# 🚀 Features

## ✅ Transformer Components Built From Scratch

- Manual implementation of Scaled Dot-Product Attention
- Custom Multi-Head Attention module
- Sinusoidal positional encoding
- Learned positional encoding
- Feed Forward Network
- Layer Normalization
- Residual Connections

> No usage of:
>
> - `torch.nn.MultiheadAttention`
> - `torch.nn.functional.multi_head_attention_forward`

---

## 📊 Streamlit Interpretability Dashboard

Interactive dashboard with:

- Attention Heatmaps
- Multi-Head Visualization
- Token Importance Analysis
- Attention Entropy Dashboard
- Layer and Head Selection Controls

---

## 📈 Training Instrumentation

Training pipeline logs:

- Attention Entropy
- Attention Weight Snapshots
- Model Checkpoints

---

## 🐳 Dockerized Deployment

The entire application is containerized using Docker and Docker Compose.

---

# 🏗️ Project Structure

```text
transformer-encoder-visualizer/
│
├── app.py
├── model.py
├── train.py
├── utils.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── .env.example
│
├── models/
│   └── final_model.pth
│
├── logs/
│   └── training_metrics.csv
│
├── snapshots/
│   ├── epoch_1_weights.pt
│   └── final_epoch_weights.pt
│
├── reports/
│   └── attention_head_biography.md
│
├── verification/
│   ├── attention_output.json
│   └── encodings_output.json
│
└── assets/
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Plotly
- Docker

---

# 🧠 Transformer Architecture

## Scaled Dot-Product Attention

The attention mechanism is implemented manually using:

```math
Attention(Q,K,V) = Softmax((QK^T) / sqrt(d_k))V
```

This allows the model to learn contextual relationships between tokens.

---

## Multi-Head Attention

The model splits embeddings into multiple heads to learn:

- syntactic relationships
- contextual dependencies
- token relevance patterns

---

## Positional Encodings

Two positional encoding methods are implemented:

### 1. Sinusoidal Encoding
Fixed mathematical encoding introduced in the original Transformer paper.

### 2. Learned Encoding
Trainable positional embeddings learned during training.

---

# 📚 Dataset

The model is trained using NLP text classification datasets.

Example:
- SST-2 Sentiment Dataset

---

# 🏋️ Training Pipeline

The training pipeline includes:

- Attention weight extraction
- Attention entropy logging
- Gradient clipping
- Learning rate warmup
- Snapshot saving

---

# 📈 Attention Entropy

Attention entropy measures how focused or diffuse each attention head is.

Low entropy:
- focused attention
- specialization

High entropy:
- distributed attention
- broad contextual understanding

Entropy values are logged in:

```text
logs/training_metrics.csv
```

---

# 🖥️ Running Locally

## 1. Clone Repository

```bash
git clone <repository-url>
cd transformer-encoder-visualizer
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Streamlit App

```bash
streamlit run app.py
```

Application runs at:

```text
http://localhost:8501
```

---

# 🐳 Docker Setup

## Build Docker Image

```bash
docker build -t transformer-visualizer .
```

---

## Run Docker Container

```bash
docker run -p 8501:8501 transformer-visualizer
```

---

## Docker Compose

```bash
docker-compose up --build
```

---

# 🔍 Verification Files

## Attention Verification

```text
verification/attention_output.json
```

Contains:

```json
{
  "input_shape": [1, 10, 128],
  "output_shape": [1, 10, 128],
  "attention_weights_shape": [1, 4, 10, 10]
}
```

---

## Positional Encoding Verification

```text
verification/encodings_output.json
```

Contains:

```json
{
  "sinusoidal_encoding_shape": [1, 20, 128],
  "learned_encoding_shape": [1, 20, 128]
}
```

---

# 📊 Streamlit Dashboard Components

## Attention Heatmap

Interactive visualization of token-to-token attention.

---

## Multi-Head Attention Explorer

Analyze individual attention heads separately.

---

## Entropy Dashboard

Visualize entropy trends across:

- layers
- heads
- epochs

---

## Token Attribution

Highlights important tokens influencing predictions.

---

# 📄 Report

Detailed interpretability analysis available at:

```text
reports/attention_head_biography.md
```

---

# 📚 References

- Attention Is All You Need (Vaswani et al.)
- The Illustrated Transformer
- The Annotated Transformer
- Streamlit Documentation

---

# 👨‍💻 Author

Transformer Encoder Visualizer Project

Built for interpretability, visualization, and deep understanding of Transformer architectures.
