# Complex_regional_systems/examples/train_multi_agent.py
import os
import sys
import yaml
import torch
from src.training.trainer import VillageTrainer
from src.utils.evaluator import VillageEvaluator
from src.utils.visualizer import VillageVisualizer

def main():
    # 加载配置
    env_config = load_config('configs/env_config.yaml')
    train_config = load_config('configs/training_config.yaml')
    
    # 创建环境和智能体
    env = VillageEnv(env_config)
    trainer = VillageTrainer(env, train_config)
    evaluator = VillageEvaluator(env, trainer.agents)
    visualizer = VillageVisualizer(env)
    
    # 训练循环
    for episode in range(train_config['num_episodes']):
        # 训练
        train_metrics = trainer.train_episode()
        
        # 定期评估
        if episode % 10 == 0:
            eval_metrics = evaluator.evaluate(num_episodes=5)
            visualizer.render()
            
        # 保存检查点
        if episode % 100 == 0:
            trainer.save_checkpoint(f"episode_{episode}")