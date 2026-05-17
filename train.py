import os
import json
import torch
import torch.nn as nn

# =========================
# SAMPLE DATA
# =========================

sentences = [
    "Transformers are powerful",
    "Encoder learns representations",
    "Attention captures context"
]

# =========================
# SIMPLE TOKENIZER
# =========================

vocab = {}
idx = 1

for sentence in sentences:
    for word in sentence.lower().split():
        if word not in vocab:
            vocab[word] = idx
            idx += 1

vocab_size = len(vocab) + 1

max_len = 6

encoded_sentences = []

for sentence in sentences:
    tokens = [vocab[word] for word in sentence.lower().split()]

    while len(tokens) < max_len:
        tokens.append(0)

    encoded_sentences.append(tokens)

input_tensor = torch.tensor(encoded_sentences)

# =========================
# MODEL
# =========================

embedding_dim = 16
num_heads = 2

embedding = nn.Embedding(vocab_size, embedding_dim)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=num_heads,
    batch_first=True
)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=2
)

# =========================
# FORWARD PASS
# =========================

embedded = embedding(input_tensor)

transformer_output = transformer_encoder(embedded)

# =========================
# SAVE REAL NUMERIC DATA
# =========================

os.makedirs("verification", exist_ok=True)

# Save embeddings
with open("verification/encodings_output.json", "w") as f:
    json.dump(
        embedded.detach().numpy().tolist(),
        f
    )

# Save transformer outputs
with open("verification/attention_output.json", "w") as f:
    json.dump(
        transformer_output.detach().numpy().tolist(),
        f
    )

print("Verification files generated successfully")