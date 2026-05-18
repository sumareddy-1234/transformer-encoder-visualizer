# 🧠 Transformer Encoder Visualizer

An interactive NLP visualization platform built with Streamlit and PyTorch to demonstrate the internal working of Transformer Encoder architectures including token embeddings, self-attention, and multi-head attention mechanisms.

---

## 🚀 Features

- 🔤 Tokenization visualization
- 🧠 Embedding heatmap visualization
- 🔥 Self-attention matrix visualization
- 🎯 Multi-head attention exploration
- 📊 Token importance analysis
- 📘 Transformer learning explanations
- 🐳 Docker support
- 📈 Training metrics logging
- 💾 Model checkpoint snapshots
- ✅ Verification output generation

---

## 🎓 Learning Objectives

This project helps users understand:

- Transformer encoder architecture
- Self-attention mechanism
- Query-Key-Value computations
- Multi-head attention
- Token relationships in NLP models
- Attention score visualization
---

## 🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- NumPy
- Pandas
- Plotly
- Docker

---

## 📂 Project Structure

```text
transformer-encoder-visualizer/
│
├── app.py
├── model.py
├── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
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
├── verification/
│   ├── attention_output.json
│   └── encodings_output.json
│
└── reports/
    └── attention_head_biography.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/sumareddy-1234/transformer-encoder-visualizer.git
cd transformer-encoder-visualizer
```

---

# ▶️ Run Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python train.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t transformer-visualizer .
```

### Run Docker Container

```bash
docker run -p 8501:8501 transformer-visualizer
```

### Run with Docker Compose

```bash
docker-compose up
```

### Rebuild Containers

```bash
docker-compose up --build
```
---

## 📊 Attention Visualization

The application visually demonstrates:

- Self-attention computation
- Query-Key-Value mechanism
- Multi-head attention
- Token importance scoring
- Embedding relationships
- Attention score distributions

---

# 🧪 Verification Outputs

Generated verification files:

- `verification/attention_output.json`
- `verification/encodings_output.json`

These files validate:
- attention tensor shapes
- positional encoding outputs
- transformer computations

---

# 📈 Training Metrics

Metrics are stored in:

```text
logs/training_metrics.csv
```

Metrics include:
- epoch number
- transformer layer
- attention head
- attention entropy

---

# 🧠 Transformer Attention Formula

The visualization demonstrates the standard transformer self-attention equation:

\[
Attention(Q, K, V) =
Softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Where:

- \(Q\) = Query matrix
- \(K\) = Key matrix
- \(V\) = Value matrix
- \(d_k\) = Dimension of key vectors

This equation computes attention scores between tokens and determines how strongly each token attends to others in the sequence.

---
# 📸 Visual Components

- Embedding heatmaps
- Attention heatmaps
- Token importance charts
- Multi-head attention explorer

---

## ✅ Generated Outputs

After execution, the project generates:

- trained model weights
- attention visualizations
- verification outputs
- training metrics
- checkpoint snapshots

---

# 👨‍💻 Author

Satti Suma
