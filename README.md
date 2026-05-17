# 🧠 Transformer Encoder Visualizer

An interactive Streamlit-based NLP visualization tool that demonstrates how Transformer Encoders work internally using token embeddings, self-attention, and multi-head attention.

---

## 🚀 Live Demo

Run the application locally:

```bash
streamlit run app.py
✨ Features
🔤 Tokenization of input sentences
🧠 Embedding generation and visualization
🔥 Self-Attention heatmap visualization
🧩 Multi-Head Attention exploration
📊 Token importance scoring
📈 Interactive Plotly heatmaps
🎛️ Streamlit dashboard UI
🌗 Dark / Light mode support
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
├── verify.py              # Testing & validation
│
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
1. Clone the repository
git clone https://github.com/sumareddy-1234/transformer-encoder-visualizer.git
cd transformer-encoder-visualizer
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
3. Install dependencies
pip install -r requirements.txt
4. Run the application
streamlit run app.py
🐳 Docker Setup
Build the Docker image
docker build -t transformer-visualizer .
Run the container
docker run -p 8501:8501 transformer-visualizer
📊 How It Works
Input sentence is tokenized
Tokens are converted into embeddings
Self-attention computes relationships between tokens
Multi-head attention captures different patterns
Attention matrices are visualized using Plotly
🎯 Learning Outcomes

This project helps you understand:

Transformer architecture in NLP
Attention mechanisms (self-attention & multi-head attention)
Word embeddings and contextual representations
Visualization of deep learning internals
How NLP models process relationships between tokens
📸 Future Improvements
Add encoder-decoder visualization
Support GPT-style attention visualization
Add real-time training visualization
Deploy on Streamlit Cloud / HuggingFace Spaces
Add BERT-style layer comparison
👨‍💻 Author

Suma Satti
