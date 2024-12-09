# Complex_regional_systems/src/models/networks.py
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64, 32]):
        super().__init__()
        layers = []
        prev_size = input_dim
        
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.LayerNorm(size)
            ])
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)