# Complex_regional_systems/src/training/replay_buffer.py
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储转换"""
        # 确保reward是标量
        if isinstance(reward, dict):
            reward = sum(float(r) if isinstance(r, (int, float)) else sum(r)
                        for r in reward.values())
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样批次数据"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)