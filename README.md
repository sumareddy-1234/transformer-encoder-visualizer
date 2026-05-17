# 🧠 Transformer Encoder Visualizer

An interactive Streamlit-based NLP visualization tool that demonstrates how Transformer Encoders work internally using token embeddings, self-attention, and multi-head attention.

---

## 🚀 Live Demo
Run locally using:
```bash
streamlit run app.py
📌 Features
🔤 Tokenization of input sentences
🧠 Embedding generation
🔥 Self-Attention visualization
🧩 Multi-Head Attention exploration
📊 Token importance scoring
📈 Interactive Plotly heatmaps
🎛️ Streamlit dashboard UI
🌗 Dark/light mode support
🏗️ Tech Stack
Python 3.11
PyTorch (CPU)
Streamlit
NumPy
Pandas
Plotly
📂 Project Structure
transformer-encoder-visualizer/
│
├── app.py                  # Streamlit frontend
├── model.py               # Transformer logic
├── train.py               # Training pipeline
├── verify.py              # Testing outputs
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
│
├── logs/
├── models/
├── snapshots/
└── verification/
⚙️ Installation
1. Clone repo
git clone https://github.com/sumareddy-1234/transformer-encoder-visualizer.git
cd transformer-encoder-visualizer
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run app
streamlit run app.py
🐳 Docker Setup
Build image
docker build -t transformer-visualizer .
Run container
docker run -p 8501:8501 transformer-visualizer
📊 How It Works
Input sentence is tokenized
Tokens converted to embeddings
Self-attention computes relationships
Multi-head attention learns different patterns
Visualizations show attention matrices
🎯 Learning Outcome

This project helps understand:

Transformer architecture
Attention mechanism
NLP representations
Visualization of deep learning models
👨‍💻 Author

suma
