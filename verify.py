import json
import torch

from model import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding
)

# -----------------------------------
# Verify Attention
# -----------------------------------

x = torch.randn(1, 10, 128)

mha = MultiHeadAttention(
    d_model=128,
    num_heads=4
)

output, attention_weights = mha(x)

attention_data = {
    "input_shape": list(x.shape),
    "output_shape": list(output.shape),
    "attention_weights_shape": list(attention_weights.shape)
}

with open(
    "verification/attention_output.json",
    "w"
) as f:

    json.dump(
        attention_data,
        f,
        indent=2
    )

# -----------------------------------
# Verify Positional Encodings
# -----------------------------------

x2 = torch.randn(1, 20, 128)

sinusoidal = SinusoidalPositionalEncoding(128)

learned = LearnedPositionalEncoding(128)

sin_output = sinusoidal(x2)

learn_output = learned(x2)

encoding_data = {
    "sinusoidal_encoding_shape": list(
        sin_output.shape
    ),
    "learned_encoding_shape": list(
        learn_output.shape
    )
}

with open(
    "verification/encodings_output.json",
    "w"
) as f:

    json.dump(
        encoding_data,
        f,
        indent=2
    )

print("Verification files created successfully!")