# Complex_regional_systems/src/training/trainer.py
import os
import torch
import numpy as np
from typing import Dict, List
import yaml
from ..utils.metrics import MetricsCalculator
from ..utils.visualizer import VillageVisualizer
from .replay_buffer import ReplayBuffer

class VillageTrainer:
    """村庄环境训练器"""
    def __init__(self, env, agents, config: Dict):
        self.env = env
        self.agents = agents
        self.config = config
            
        # 初始化组件
        self.metrics = MetricsCalculator()
        self.visualizer = VillageVisualizer(env)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # 初始化优化器
        self.optimizers = {}
        for agent_type, agent in self.agents.items():
            if isinstance(agent, list):
                # 如果是智能体列表（如entrepreneurs和laborers）
                self.optimizers[agent_type] = [
                    torch.optim.Adam(a.parameters(), lr=self.config['training']['learning_rate'])
                    for a in agent
                ]
            else:
                # 如果是单个智能体（如leader）
                self.optimizers[agent_type] = torch.optim.Adam(
                    agent.parameters(),
                    lr=self.config['training']['learning_rate']
                )
        
        # 训练状态
        self.episode = 0
        self.global_step = 0
        self.best_reward = float('-inf')
        
        # 创建保存目录
        self.save_dir = os.path.join('results', 'training')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_episode(self) -> Dict:
        """训练一个episode"""
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        transitions = []
        
        while not done:
            # 收集智能体动作
            actions = {}
            for agent_type, agent in self.agents.items():
                if isinstance(agent, list):
                    # 如果是智能体列表（如entrepreneurs和laborers）
                    actions[agent_type] = [a.act(state) for a in agent]
                else:
                    # 如果是单个智能体（如leader）
                    actions[agent_type] = agent.act(state)
            
            # 环境步进
            next_state, rewards, done, info = self.env.step(actions)
            
            # 计算总奖励
            total_reward = 0.0
            for agent_type, reward in rewards.items():
                if isinstance(reward, list):
                    total_reward += sum(float(r) for r in reward)
                else:
                    total_reward += float(reward)
            
            # 存储转换
            self.replay_buffer.push(state, actions, rewards, next_state, done)
            
            # 更新状态
            state = next_state
            episode_reward += total_reward
            self.global_step += 1
            
            # 如果积累了足够的转换,就进行学习
            if len(self.replay_buffer) >= self.config['training']['batch_size']:
                self._learn()
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'steps': self.global_step
        }
    
    def _learn(self):
        """从经验回放中学习"""
        # 采样批次数据
        batch = self.replay_buffer.sample(self.config['training']['batch_size'])
        
        # 对每个智能体进行学习
        for agent_type, agent in self.agents.items():
            if isinstance(agent, list):
                # 如果是智能体列表
                for i, a in enumerate(agent):
                    # 计算损失
                    loss = a.compute_loss(batch)
                    
                    # 优化
                    self.optimizers[agent_type][i].zero_grad()
                    loss.backward()
                    self.optimizers[agent_type][i].step()
            else:
                # 如果是单个智能体
                loss = agent.compute_loss(batch)
                
                # 优化
                self.optimizers[agent_type].zero_grad()
                loss.backward()
                self.optimizers[agent_type].step()
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'episode': self.episode,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'agent_states': {
                agent_type: agent.state_dict()
                for agent_type, agent in self.agents.items()
            },
            'optimizer_states': {
                agent_type: opt.state_dict()
                for agent_type, opt in self.optimizers.items()
            }
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_{name}.pt')
        torch.save(checkpoint, path)