import os
import json
import math
import torch
import pandas as pd

from model import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    TransformerEncoder
)

# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("verification", exist_ok=True)

# =========================================================
# ATTENTION VERIFICATION
# =========================================================

batch_size = 1
seq_len = 10
d_model = 128
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)

attention = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads
)

output, attention_weights = attention(x)

attention_verification = {
    "input_shape": list(x.shape),
    "output_shape": list(output.shape),
    "attention_weights_shape": list(attention_weights.shape)
}

with open("verification/attention_output.json", "w") as f:
    json.dump(attention_verification, f, indent=4)

# =========================================================
# POSITIONAL ENCODING VERIFICATION
# =========================================================

encoding_input = torch.randn(1, 20, 128)

sinusoidal = SinusoidalPositionalEncoding(
    d_model=128,
    max_len=20
)

learned = LearnedPositionalEncoding(
    d_model=128,
    max_len=20
)

sin_output = sinusoidal(encoding_input)
learned_output = learned(encoding_input)

encoding_verification = {
    "sinusoidal_encoding_shape": list(sin_output.shape),
    "learned_encoding_shape": list(learned_output.shape)
}

with open("verification/encodings_output.json", "w") as f:
    json.dump(encoding_verification, f, indent=4)

# =========================================================
# CREATE MODEL
# =========================================================

model = TransformerEncoder(
    vocab_size=1000,
    d_model=128,
    num_heads=4,
    num_layers=2,
    max_len=100,
    num_classes=2
)

# =========================================================
# DUMMY TRAINING LOOP
# =========================================================

metrics = []

epochs = 5

for epoch in range(1, epochs + 1):

    dummy_input = torch.randint(
        0,
        1000,
        (1, 10)
    )

    outputs, attention_list = model(dummy_input)

    # =====================================================
    # ENTROPY CALCULATION
    # =====================================================

    for layer_idx, layer_attention in enumerate(attention_list):

        for head_idx in range(layer_attention.shape[1]):

            attn = layer_attention[0, head_idx]

            entropy = -torch.sum(
                attn * torch.log(attn + 1e-9)
            ).item()

            metrics.append({
                "epoch": epoch,
                "layer": layer_idx,
                "head": head_idx,
                "attention_entropy": entropy
            })

    # =====================================================
    # SAVE SNAPSHOTS
    # =====================================================

    if epoch == 1:
        torch.save(
            attention_list,
            "snapshots/epoch_1_weights.pt"
        )

    if epoch == epochs:
        torch.save(
            attention_list,
            "snapshots/final_epoch_weights.pt"
        )

# =========================================================
# SAVE METRICS
# =========================================================

df = pd.DataFrame(metrics)

df.to_csv(
    "logs/training_metrics.csv",
    index=False
)

# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "models/final_model.pth"
)

print("Training completed successfully.")