# 🧠 Transformer Encoder Visualizer

An interactive Streamlit-based NLP visualization tool that demonstrates how Transformer Encoders work internally using token embeddings, self-attention, and multi-head attention.

---

# 🚀 Features

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

# 🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- NumPy
- Pandas
- Plotly
- Docker

---

# 📂 Project Structure

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
git clone <repository-url>
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

# 🐳 Docker Setup

## Build Docker Image

```bash
docker build -t transformer-visualizer .
```

## Run Container

```bash
docker run -p 8501:8501 transformer-visualizer
```

---

# 📊 Attention Visualization

The application demonstrates:

- Self-attention computation
- Query-Key-Value mechanism
- Multi-head attention
- Token importance scoring
- Embedding visualization

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

# 🧠 Transformer Formula

The visualization demonstrates the standard transformer attention equation:

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

---

# 📸 Visual Components

- Embedding heatmaps
- Attention heatmaps
- Token importance charts
- Multi-head attention explorer

---

# ✅ Output Files

After execution the project generates:

- trained model weights
- attention snapshots
- verification outputs
- training metrics

---

# 👨‍💻 Author

Transformer Encoder Visualizer Project
