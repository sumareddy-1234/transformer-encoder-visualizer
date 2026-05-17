import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Transformer Encoder Visualizer",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "run_model" not in st.session_state:
    st.session_state.run_model = False

# =========================================================
# MODERN COLOR PALETTE
# =========================================================

st.markdown("""
<style>

/* =========================================================
GLOBAL
========================================================= */

html, body, [class*="css"]  {
    font-family: "Inter", sans-serif;
}

/* =========================================================
LIGHT MODE
========================================================= */

.stApp {
    background:
        radial-gradient(circle at top left, #eef4ff 0%, transparent 30%),
        radial-gradient(circle at top right, #f3e8ff 0%, transparent 30%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);

    color: #0f172a;
}

/* =========================================================
DARK MODE
========================================================= */

@media (prefers-color-scheme: dark) {

    .stApp {

        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.15) 0%, transparent 25%),
            radial-gradient(circle at top right, rgba(139,92,246,0.12) 0%, transparent 25%),
            linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;

        color: #f8fafc !important;
    }

    p,
    span,
    div,
    label {
        color: #e2e8f0 !important;
    }

    h1 {
        color: #93c5fd !important;
    }

    h2, h3, h4 {
        color: #f8fafc !important;
    }
}

/* =========================================================
HEADINGS
========================================================= */

h1 {

    font-size: 3rem !important;

    font-weight: 800 !important;

    color: #2563eb !important;

    letter-spacing: -1px;
}

h2, h3 {

    font-weight: 700 !important;

    color: #0f172a !important;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background: rgba(255,255,255,0.72);

    backdrop-filter: blur(18px);

    border-right: 1px solid rgba(148,163,184,0.25);
}

@media (prefers-color-scheme: dark) {

    section[data-testid="stSidebar"] {

        background: rgba(15,23,42,0.75) !important;

        backdrop-filter: blur(20px);

        border-right: 1px solid rgba(51,65,85,0.5) !important;
    }
}

/* =========================================================
TEXT AREA + INPUTS
========================================================= */

textarea,
input,
.stTextArea textarea,
.stTextInput input {

    background: rgba(255,255,255,0.88) !important;

    border: 1px solid #cbd5e1 !important;

    border-radius: 14px !important;

    color: #0f172a !important;

    transition: 0.3s ease;
}

textarea:focus,
input:focus {

    border: 1px solid #3b82f6 !important;

    box-shadow: 0 0 0 3px rgba(59,130,246,0.18);
}

@media (prefers-color-scheme: dark) {

    textarea,
    input,
    .stTextArea textarea,
    .stTextInput input {

        background: rgba(30,41,59,0.85) !important;

        border: 1px solid #334155 !important;

        color: #f8fafc !important;
    }
}

/* =========================================================
SLIDER
========================================================= */

.stSlider {

    padding-top: 12px;

    padding-bottom: 12px;
}

/* =========================================================
BUTTON
========================================================= */

.stButton button {

    width: 100%;

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    color: white !important;

    border: none;

    border-radius: 14px;

    font-weight: 700;

    padding: 0.75rem 1rem;

    box-shadow: 0 8px 25px rgba(37,99,235,0.25);

    transition: all 0.25s ease;
}

.stButton button:hover {

    transform: translateY(-2px);

    box-shadow: 0 10px 28px rgba(99,102,241,0.35);
}

/* =========================================================
TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 10px;

    margin-bottom: 10px;
}

.stTabs [data-baseweb="tab"] {

    background: rgba(255,255,255,0.7);

    border-radius: 14px;

    border: 1px solid rgba(203,213,225,0.7);

    padding: 12px 20px;

    font-weight: 600;

    color: #0f172a;

    transition: 0.25s ease;
}

.stTabs [aria-selected="true"] {

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    border: none !important;
}

@media (prefers-color-scheme: dark) {

    .stTabs [data-baseweb="tab"] {

        background: rgba(30,41,59,0.7) !important;

        border: 1px solid rgba(51,65,85,0.7) !important;

        color: #e2e8f0 !important;
    }
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card {

    background: linear-gradient(
        135deg,
        rgba(37,99,235,0.95),
        rgba(124,58,237,0.95)
    );

    padding: 20px;

    border-radius: 18px;

    color: white;

    text-align: center;

    box-shadow: 0 10px 30px rgba(59,130,246,0.22);

    backdrop-filter: blur(14px);
}

.metric-card h4 {

    color: white !important;
}

/* =========================================================
INFO BOX
========================================================= */

.info-box {

    background: rgba(255,255,255,0.72);

    border-left: 5px solid #2563eb;

    padding: 18px;

    border-radius: 16px;

    backdrop-filter: blur(14px);

    color: #0f172a;

    box-shadow: 0 4px 18px rgba(15,23,42,0.06);
}

@media (prefers-color-scheme: dark) {

    .info-box {

        background: rgba(30,41,59,0.65) !important;

        border-left: 5px solid #60a5fa !important;

        color: #e2e8f0 !important;

        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    border-radius: 16px;

    overflow: hidden;

    border: 1px solid rgba(203,213,225,0.25);
}

/* =========================================================
PLOTLY
========================================================= */

.js-plotly-plot {

    border-radius: 18px;

    overflow: hidden;

    background: transparent !important;
}

/* =========================================================
METRICS
========================================================= */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.75);

    border: 1px solid rgba(203,213,225,0.4);

    padding: 18px;

    border-radius: 18px;

    backdrop-filter: blur(16px);
}

@media (prefers-color-scheme: dark) {

    [data-testid="metric-container"] {

        background: rgba(30,41,59,0.72) !important;

        border: 1px solid rgba(51,65,85,0.5) !important;
    }
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# TITLE
# =========================================================

st.title("🧠 Transformer Encoder Visualizer")

st.markdown("""
### Interactive NLP Dashboard for Understanding Transformer Encoders

Explore:

- 🔤 Token Embeddings
- 🔥 Self Attention
- 🧠 Multi-Head Attention
- 📊 Token Importance
- 📘 Transformer Learning
""")

# =========================================================
# TOKENIZER
# =========================================================

def tokenize(text):

    if not text:
        return []

    tokens = text.lower().strip().split()

    cleaned = []

    for token in tokens:

        token = token.strip(".,!?;:()[]{}")

        if token:
            cleaned.append(token)

    return cleaned

# =========================================================
# EMBEDDING
# =========================================================

def embedding(token, dim):

    seed = sum(ord(c) for c in token)

    rng = np.random.default_rng(seed)

    return rng.normal(0, 1, dim)

# =========================================================
# SOFTMAX
# =========================================================

def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)

    exp = np.exp(x)

    return exp / np.sum(exp, axis=-1, keepdims=True)

# =========================================================
# MULTI HEAD ATTENTION
# =========================================================

def multi_head_attention(X, num_heads):

    n, d = X.shape

    num_heads = min(num_heads, d)

    while d % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    head_dim = d // num_heads

    attention_heads = []

    outputs = []

    for i in range(num_heads):

        start = i * head_dim
        end = start + head_dim

        X_head = X[:, start:end]

        rng = np.random.default_rng(i + 50)

        Wq = rng.normal(size=(head_dim, head_dim))
        Wk = rng.normal(size=(head_dim, head_dim))
        Wv = rng.normal(size=(head_dim, head_dim))

        Q = X_head @ Wq
        K = X_head @ Wk
        V = X_head @ Wv

        scores = (Q @ K.T) / np.sqrt(head_dim)

        weights = softmax(scores)

        output = weights @ V

        attention_heads.append(weights)

        outputs.append(output)

    final_output = np.concatenate(outputs, axis=1)

    return attention_heads, final_output, num_heads

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Controls")

sentence = st.sidebar.text_area(
    "Enter Sentence",
    value="Transformers understand relationships between words using attention",
    height=120
)

embed_dim = st.sidebar.slider(
    "Embedding Size",
    min_value=4,
    max_value=32,
    value=8,
    step=4
)

num_heads = st.sidebar.selectbox(
    "Attention Heads",
    options=[1, 2, 4, 8],
    index=1
)

show_math = st.sidebar.toggle(
    "Show Formula",
    value=True
)

# =========================================================
# RUN BUTTON
# =========================================================

if st.sidebar.button("🚀 Run Visualization"):
    st.session_state.run_model = True

# =========================================================
# INITIAL SCREEN
# =========================================================

if not st.session_state.run_model:

    st.info("👈 Configure controls from the sidebar and click Run Visualization")

    st.stop()

# =========================================================
# TOKENS
# =========================================================

tokens = tokenize(sentence)

if len(tokens) < 2:

    st.error("Please enter at least 2 words.")

    st.stop()

# =========================================================
# UNIQUE TOKENS
# =========================================================

counter = Counter()

unique_tokens = []

for token in tokens:

    counter[token] += 1

    if counter[token] == 1:
        unique_tokens.append(token)
    else:
        unique_tokens.append(f"{token}_{counter[token]}")

# =========================================================
# EMBEDDINGS
# =========================================================

X = np.array([
    embedding(token, embed_dim)
    for token in tokens
])

# =========================================================
# ATTENTION
# =========================================================

heads, encoder_output, actual_heads = multi_head_attention(
    X,
    num_heads
)

avg_attention = np.mean(heads, axis=0)

# =========================================================
# METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Tokens", len(tokens))
m2.metric("Embedding Size", embed_dim)
m3.metric("Attention Heads", actual_heads)
m4.metric("Sequence Length", len(tokens))

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Sentence",
    "🔤 Embeddings",
    "🔥 Attention",
    "🧠 Multi-Head",
    "📊 Insights",
    "📘 Learn"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Sentence Breakdown")

    token_df = pd.DataFrame({
        "Position": range(len(tokens)),
        "Token": tokens,
        "Characters": [len(t) for t in tokens]
    })

    st.dataframe(token_df, use_container_width=True)

    cols = st.columns(len(tokens))

    for i, token in enumerate(tokens):

        cols[i].markdown(
            f"""
            <div class="metric-card">
                <h4>{token}</h4>
                <p>Position {i}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class="info-box">
    The transformer first converts the sentence into tokens.
    Each token becomes a numerical representation called an embedding.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Embedding Space")

    fig = px.imshow(
        X,
        x=[f"D{i}" for i in range(embed_dim)],
        y=unique_tokens,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
    <div class="info-box">
    Hover over the embedding heatmap to inspect vector values interactively.
    Similar words often generate similar patterns.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("Self Attention Matrix")

    fig = px.imshow(
        avg_attention,
        x=unique_tokens,
        y=unique_tokens,
        text_auto=".2f",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Inspect Attention for a Token")

    selected_token = st.radio(
        "Choose Token",
        unique_tokens,
        horizontal=True
    )

    token_index = unique_tokens.index(selected_token)

    token_attention = avg_attention[token_index]

    token_df = pd.DataFrame({
        "Related Token": unique_tokens,
        "Attention Score": token_attention
    })

    token_df = token_df.sort_values(
        "Attention Score",
        ascending=False
    )

    bar_fig = px.bar(
        token_df,
        x="Related Token",
        y="Attention Score",
        color="Attention Score",
        color_continuous_scale="Teal"
    )

    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True
    )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader("Multi-Head Attention Explorer")

    # =====================================================
    # SESSION STATE FOR HEAD SELECTION
    # =====================================================

    if "selected_head" not in st.session_state:
        st.session_state.selected_head = 1

    # RESET IF OUT OF RANGE
    if st.session_state.selected_head > actual_heads:
        st.session_state.selected_head = 1

    # =====================================================
    # DROPDOWN
    # =====================================================

    selected_head = st.selectbox(
        "Select Attention Head",
        options=list(range(1, actual_heads + 1)),
        key="selected_head"
    )

    # =====================================================
    # CURRENT HEAD MATRIX
    # =====================================================

    head_matrix = heads[selected_head - 1]

    # =====================================================
    # HEATMAP
    # =====================================================

    fig = px.imshow(
        head_matrix,
        x=unique_tokens,
        y=unique_tokens,
        text_auto=".2f",
        color_continuous_scale="Purples",
        aspect="auto"
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # INFO BOX
    # =====================================================

    st.markdown(
        f"""
        <div class="info-box">
        <b>Currently Visualizing Attention Head {selected_head}</b><br><br>
        Each attention head learns different contextual relationships between tokens.
        Changing the dropdown updates the visualization interactively.
        </div>
        """,
        unsafe_allow_html=True
    )
# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader("Token Importance")

    importance = avg_attention.mean(axis=1)

    imp_df = pd.DataFrame({
        "Token": unique_tokens,
        "Importance": importance
    })

    imp_df = imp_df.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        imp_df,
        use_container_width=True
    )

    fig = px.bar(
        imp_df,
        x="Token",
        y="Importance",
        color="Importance",
        color_continuous_scale="Sunset"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# TAB 6
# =========================================================

with tab6:

    st.subheader("How Transformer Encoders Work")

    st.markdown("""
    ### 1️⃣ Tokenization
    The sentence is split into tokens.

    ### 2️⃣ Embeddings
    Tokens become vector representations.

    ### 3️⃣ Self Attention
    Tokens learn which words matter most.

    ### 4️⃣ Multi-Head Attention
    Different heads learn different relationships.

    ### 5️⃣ Encoder Output
    Final context-aware representations are created.
    """)

    if show_math:

        st.markdown("## Attention Formula")

        st.latex(r'''
        Attention(Q,K,V)
        =
        Softmax
        \left(
        \frac{QK^T}{\sqrt{d}}
        \right)V
        ''')

        st.markdown("""
        ### Formula Explanation

        - **Q (Query)** → What the token is searching for
        - **K (Key)** → What information a token provides
        - **V (Value)** → Information passed to the next layer
        - **Softmax** → Converts scores into probabilities
        - **√d** → Scaling factor for stable learning
        """)
# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.success("Visualization completed successfully.")

st.caption(
    "Transformer Encoder Visualizer | Interactive NLP Dashboard"
)