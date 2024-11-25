import torch
import torch.nn as nn

# Individual input shape (4, 80, 80)
x_individual = torch.randn(4, 80, 80)

# Batched input shape (32, 4, 80, 80)
x_batched = torch.randn(32, 4, 80, 80)

# Create a Sequential model with a Flatten layer
flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)  # Flatten from the second dimension to the second-to-last dimension
output_individual = flatten_layer(x_individual)
output_batched = flatten_layer(x_batched)

# Check the output dimensions
print("Output dimension for individual input:", output_individual.shape)
print("Output dimension for batched input:", output_batched.shape)


