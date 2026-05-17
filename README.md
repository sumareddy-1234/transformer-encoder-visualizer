🧠 Transformer Encoder Visualizer
https://img.shields.io/badge/Python-3.11-blue?logo=python  
https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch  
https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit  
https://img.shields.io/badge/NumPy-Numerical%20Computing-yellow?logo=numpy  
https://img.shields.io/badge/Pandas-Data%20Analysis-lightblue?logo=pandas  
https://img.shields.io/badge/Plotly-Interactive%20Charts-green?logo=plotly

📌 Project Overview
The Transformer Encoder Visualizer is an interactive NLP visualization tool built with Streamlit.
It demonstrates how Transformer Encoders work internally — from token embeddings to self-attention and multi-head attention — with intuitive visualizations.

✨ Features
Tokenization of input sentences

Embedding generation and visualization

Self-Attention heatmap visualization

Multi-Head Attention exploration

Token importance scoring

Interactive Plotly heatmaps

Streamlit dashboard UI with dark/light mode

📂 Project Structure
plaintext
transformer-encoder-visualizer/
│
├── app.py                  # Streamlit frontend
├── model.py                # Transformer logic
├── train.py                # Training pipeline
├── verify.py               # Testing & validation
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
Clone repo

bash
git clone https://github.com/sumareddy-1234/transformer-encoder-visualizer.git
cd transformer-encoder-visualizer
Create venv

bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
Install dependencies

bash
pip install -r requirements.txt
Run app

bash
streamlit run app.py
🐳 Docker Setup
Build image

bash
docker build -t transformer-visualizer .
Run container

bash
docker run -p 8501:8501 transformer-visualizer
📊 How It Works
Input sentence is tokenized

Tokens → embeddings

Self-attention learns relationships

Multi-head attention captures patterns

Attention maps visualized using Plotly

🎯 Learning Outcomes
Transformer architecture

Self-attention mechanism

Multi-head attention

Word embeddings

NLP visualization concepts

👨‍💻 Author
Suma Satti
