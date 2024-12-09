# Complex_regional_systems/src/models/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # 隐藏层
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layers[0](x)))
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))
        mean = torch.tanh(self.layers[-1](x))
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    """价值网络"""
    def __init__(self, input_dim, hidden_sizes=[256, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        
        # 隐藏层
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layers[0](x)))
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)