# Complex_regional_systems/src/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns

class ExperimentVisualizer:
    """实验结果可视化"""
    def __init__(self):
        plt.style.use('seaborn')
        self.fig_size = (15, 12)
        
    def plot_comparison(self, experiments: List[Dict]):
        """比较多个实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        
        # 绘制学习曲线
        self._plot_learning_curves(axes[0,0], experiments)
        
        # 绘制资源利用率
        self._plot_resource_usage(axes[0,1], experiments)
        
        # 绘制社会指标
        self._plot_social_metrics(axes[1,0], experiments)
        
        # 绘制经济指标
        self._plot_economic_metrics(axes[1,1], experiments)
        
        plt.tight_layout()
        return fig
    
    def _plot_learning_curves(self, ax, experiments):
        """绘制学习曲线"""
        for exp in experiments:
            rewards = [step['reward'] for step in exp['results']]
            ax.plot(rewards, label=exp['name'])
        
        ax.set_title('Learning Curves')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        
        # 添加移动平均线
        window = 100
        for exp in experiments:
            rewards = [step['reward'] for step in exp['results']]
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg, '--', alpha=0.5)
    
    def _plot_resource_usage(self, ax, experiments):
        """绘制资源利用率"""
        for exp in experiments:
            resource_usage = [step['resource_usage'] for step in exp['results']]
            ax.plot(resource_usage, label=f"{exp['name']}")
        
        ax.set_title('Resource Utilization')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Resource Usage Rate')
        ax.legend()
    
    def _plot_social_metrics(self, ax, experiments):
        """绘制社会指标"""
        metrics = ['gini', 'social_welfare', 'cooperation']
        
        for exp in experiments:
            data = []
            for metric in metrics:
                values = [step[metric] for step in exp['results']]
                data.append(np.mean(values))
            
            ax.bar(np.arange(len(metrics)) + 0.2 * experiments.index(exp),
                  data, width=0.2, label=exp['name'])
        
        ax.set_title('Social Metrics')
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.legend()
    
    def _plot_economic_metrics(self, ax, experiments):
        """绘制经济指标"""
        for exp in experiments:
            gdp = [step['gdp'] for step in exp['results']]
            ax.plot(gdp, label=f"{exp['name']} GDP")
            
            # 添加趋势线
            z = np.polyfit(range(len(gdp)), gdp, 1)
            p = np.poly1d(z)
            ax.plot(p(range(len(gdp))), '--', alpha=0.5)
        
        ax.set_title('Economic Growth')
        ax.set_xlabel('Episode')
        ax.set_ylabel('GDP')
        ax.legend()
    
    def save_plots(self, experiments: List[Dict], path: str):
        """保存所有图表"""
        fig = self.plot_comparison(experiments)
        fig.savefig(path)
        plt.close(fig)