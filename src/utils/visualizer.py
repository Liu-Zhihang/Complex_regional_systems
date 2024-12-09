# Complex_regional_systems/src/utils/visualizer.py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Dict, List
import seaborn as sns
from datetime import datetime
import imageio
import traceback

class VillageVisualizer:
    """实验结果可视化"""
    def __init__(self, env):
        self.env = env
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        plt.ion()  # 开启交互模式
        
        # 创建保存目录
        self.save_dir = os.path.join('results', 'visualizations', 
                                    datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 存储图像列表而不是frames
        self.images = []
        
        # 颜色映射
        self.agent_colors = {
            'leader': 'red',
            'entrepreneur': 'blue',
            'laborer': 'green'
        }
        
        # 历史数据
        self.history = {
            'social_welfare': [],
            'resource_levels': [],
            'wealth_distribution': [],
            'skill_levels': []
        }
    
    def render(self, episode: int, step: int):
        """渲染并保存当前状态"""
        try:
            # 清除当前图像
            for ax in self.axes.flat:
                ax.clear()
            
            # 绘制各个子图
            self._plot_terrain_and_resources(self.axes[0,0])
            self._plot_agents(self.axes[0,1])
            self._plot_economic_metrics(self.axes[1,0])
            self._plot_social_metrics(self.axes[1,1])
            
            plt.tight_layout()
            
            # 保存当前图像
            filename = f'episode_{episode:03d}_step_{step:03d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            
            # 将图像添加到列表
            self.images.append(filepath)
            
            # 实时显示
            plt.draw()
            plt.pause(0.1)
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
    
    def save_animation(self, filename='simulation.gif'):
        """保存GIF动画"""
        try:
            # 读取所有保存的图像
            images = []
            for filepath in self.images:
                images.append(imageio.imread(filepath))
            
            # 保存为GIF
            output_path = os.path.join(self.save_dir, filename)
            imageio.mimsave(output_path, images, duration=0.5)  # 每帧持续0.5秒
            print(f"Animation saved to {output_path}")
            
        except Exception as e:
            print(f"Animation save error: {str(e)}")
            traceback.print_exc()
    
    def save_history(self):
        """保存历史数据"""
        history_file = os.path.join(self.save_dir, 'history.npz')
        np.savez(history_file, **self.history)
    
    def _plot_terrain_and_resources(self, ax):
        """绘制地形和资源"""
        ax.clear()
        
        # 绘制地形
        terrain = self.env.space_system.terrain
        if np.any(np.isfinite(terrain)):  # 检查数据是否有效
            ax.imshow(terrain, cmap='terrain', alpha=0.5)
        
        # 叠加资源分布
        for resource_type in self.env.resource_system.resources:
            amount = self.env.resource_system.resources[resource_type]['amount']
            if np.any(np.isfinite(amount)):  # 检查数据是否有效
                if resource_type == 'water':
                    ax.imshow(amount, cmap='Blues', alpha=0.3)
                elif resource_type == 'forest':
                    ax.imshow(amount, cmap='Greens', alpha=0.3)
                else:
                    ax.imshow(amount, cmap='YlOrBr', alpha=0.3)
        
        ax.set_title('Terrain & Resources')
    
    def _plot_agents(self, ax):
        """绘制智能体"""
        ax.clear()
        
        # 绘制地形作为背景
        if np.any(np.isfinite(self.env.space_system.terrain)):
            ax.imshow(self.env.space_system.terrain, cmap='terrain', alpha=0.3)
        
        # 绘制智能体
        for agent_type, positions in self.env.space_system.agent_positions.items():
            if isinstance(positions, list):
                for pos in positions:
                    self._draw_agent(ax, pos, agent_type)
            else:
                self._draw_agent(ax, positions, agent_type)
        
        ax.set_title('Agents')
    
    def _plot_economic_metrics(self, ax):
        """绘制经济指标"""
        ax.clear()
        
        # 获取财富数据
        wealth_data = []
        if hasattr(self.env, 'agent_states'):
            for agent_type, states in self.env.agent_states.items():
                if isinstance(states, list):
                    wealth_data.extend([s['wealth'] for s in states if np.isfinite(s['wealth'])])
                elif isinstance(states, dict) and 'wealth' in states:
                    if np.isfinite(states['wealth']):
                        wealth_data.append(states['wealth'])
        
        if wealth_data:  # 只在有有效数据时绘图
            ax.plot(range(len(wealth_data)), sorted(wealth_data), label='Wealth Distribution')
            ax.set_title('Economic Status')
            ax.set_xlabel('Agent Rank')
            ax.set_ylabel('Wealth')
            ax.legend()
    
    def _plot_social_metrics(self, ax):
        """绘制社会指标"""
        ax.clear()
        
        # 获取财富数据
        all_wealth = []
        if hasattr(self.env, 'agent_states'):
            for states in self.env.agent_states.values():
                if isinstance(states, list):
                    all_wealth.extend([s['wealth'] for s in states if np.isfinite(s['wealth'])])
                elif isinstance(states, dict) and 'wealth' in states:
                    if np.isfinite(states['wealth']):
                        all_wealth.append(states['wealth'])
        
        if all_wealth:  # 只在有有效数据时绘图
            ax.hist(all_wealth, bins=20, alpha=0.7)
            ax.set_title('Wealth Distribution')
            ax.set_xlabel('Wealth')
            ax.set_ylabel('Count')
    
    def _draw_agent(self, ax, position, agent_type):
        """绘制单个智能体"""
        if position is not None and len(position) == 2:
            color = self.agent_colors.get(agent_type.split('_')[0], 'gray')
            ax.scatter(position[1], position[0], c=color, alpha=0.7)