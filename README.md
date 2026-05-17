🧠 Transformer Encoder Visualizer

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit) ![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-yellow?logo=numpy) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightblue?logo=pandas) ![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-green?logo=plotly)

The Transformer Encoder Visualizer is an interactive NLP visualization tool built with Streamlit. It demonstrates how Transformer Encoders work internally — from token embeddings to self-attention and multi-head attention — with intuitive Plotly-based visualizations and an interactive dashboard.

Badges

The repository includes status/technology badges at the top (rendered inline in the title section).

✨ Features

- Tokenization of input sentences
- Embedding generation and visualization
- Self-attention heatmap visualization
- Multi-head attention exploration
- Token importance scoring
- Interactive Plotly heatmaps
- Streamlit dashboard UI with dark/light mode

Tech Stack

- Python 3.11
- PyTorch
- Streamlit
- NumPy
- Pandas
- Plotly

(These are reflected by the badges and requirements.txt.)

📂 Project Structure

Use a fenced code block for the tree to preserve formatting:

```
transformer-encoder-visualizer/
├── app.py                  # Streamlit frontend
├── model.py                # Transformer logic
├── train.py                # Training pipeline
├── verify.py               # Testing & validation
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
├── logs/
├── models/
├── snapshots/
└── verification/
```

Add brief descriptions for non-obvious directories (e.g., models/ stores saved models, snapshots/ stores example visual outputs).

⚙️ Installation

1) Clone the repository

```bash
git clone https://github.com/sumareddy-1234/transformer-encoder-visualizer.git
cd transformer-encoder-visualizer
```

2) Create and activate a virtual environment

- Windows

```bash
python -m venv venv
venv\Scripts\activate
```

- macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:
- If you encounter dependency issues, consider using pip-tools or poetry.
- Provide a requirements.txt or lockfile to ensure reproducible installs.

Run Locally

After installing dependencies and activating the virtual environment, start the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit (typically http://localhost:8501) to use the interactive visualizer.

Example quick test:
- In the app input box, enter: "The quick brown fox jumps over the lazy dog"
- Toggle attention heads and observe the Plotly heatmaps and token importance scores.

🐳 Docker Setup

Build the Docker image:

```bash
docker build -t transformer-visualizer .
```

Run the container (expose Streamlit default port 8501):

```bash
docker run -p 8501:8501 transformer-visualizer
```

Consider adding a docker-compose.yml for development and specifying a non-root user and reduced image size (e.g., use a slim Python base image) for production use.

Usage / Examples

How it works (high level):
1. Input sentence is tokenized.
2. Tokens are mapped to embeddings.
3. Self-attention computes relationships and attention weights.
4. Multi-head attention captures different relational patterns.
5. Attention maps are shown with interactive Plotly heatmaps.

Example (what to expect in the UI):
- Enter: "Transformers are powerful for NLP tasks." → tokens and embeddings appear.
- Click an attention head to see a heatmap of token-to-token attention scores.
- Use dark/light toggle to change theme.

Contributing

Thank you for your interest in contributing! A minimal contributing guide:

- Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
- Run tests (if any) and ensure linting passes.
- Open a Pull Request describing the change and link to any related issue.

Add a CONTRIBUTING.md file to this repository with repo-specific testing and style rules.

Running Tests

The original README references `verify.py` but does not include instructions. Add tests and run them with a command similar to:

```bash
python -m pytest tests/
```

Or provide `verify.py` usage details, e.g.: `python verify.py --sample tests/sample_input.txt`.

Environment Variables

If the app requires any secrets, API keys, or configurable parameters, document them here and provide an `.env.example`. Example:

```
# .env.example
STREAMLIT_SERVER_PORT=8501
MODEL_PATH=models/latest.pt
```

Load env vars via python-dotenv or Streamlit secrets as appropriate.

License

The original README did not include a license. Add a LICENSE file (e.g., MIT) and include a one-line license notice here, for example:

MIT License — see LICENSE file.

If you intend a different license, specify it and include the LICENSE file in the repo.

👨‍💻 Author

Suma Satti

Contact: (add an email or GitHub profile link here)

Contributions and bug reports are welcome — please open an issue or PR.

Acknowledgements

- Inspiration: Transformer papers and Streamlit examples
- Libraries: PyTorch, Streamlit, Plotly, NumPy, Pandas

Consider adding links to key resources (original Transformer paper, Streamlit docs, Plotly docs).

Demo

If you have a live demo (e.g., via Streamlit sharing, Heroku, or GitHub Pages), add the link here. Consider adding a screenshot or short GIF showing the UI and an example visualization.

Example placeholder:

![screenshot](docs/screenshot.png)

(Include `docs/` with images in the repo.)
