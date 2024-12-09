# Complex_regional_systems/src/utils/analysis.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class TrainingAnalyzer:
    """训练结果分析器"""
    def __init__(self, metrics_history: Dict):
        self.history = metrics_history
        
    def plot_learning_curves(self):
        """绘制学习曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制奖励曲线
        self._plot_rewards(axes[0,0])
        # 绘制社会福利
        self._plot_social_welfare(axes[0,1])
        # 绘制资源可持续性
        self._plot_sustainability(axes[1,0])
        # 绘制基尼系数
        self._plot_gini(axes[1,1])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_rewards(self, ax):
        rewards = self.history['rewards']
        ax.plot(rewards, label='Total Reward')
        ax.set_title('Training Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()