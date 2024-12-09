# Complex_regional_systems/src/models/losses.py
import torch
import torch.nn.functional as F

class PolicyGradientLoss:
    """策略梯度损失"""
    def __call__(self, logits, actions, rewards):
        policy_loss = -torch.mean(torch.log(logits) * rewards)
        return policy_loss

class ValueLoss:
    """价值网络损失"""
    def __call__(self, predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values)