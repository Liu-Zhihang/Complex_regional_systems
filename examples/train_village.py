# Complex_regional_systems/examples/train_village.py
import os
import sys
import yaml
from src.training.trainer import VillageTrainer
from src.environment import VillageEnv
from src.agents import LeaderAgent, EntrepreneurAgent, LaborAgent

def main():
    # 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
    with open('configs/env_config.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
        
    # 创建环境和智能体
    env = VillageEnv(env_config)
    agents = {
        'leader': LeaderAgent(env_config),
        'entrepreneurs': [EntrepreneurAgent(env_config) for _ in range(10)],
        'laborers': [LaborAgent(env_config) for _ in range(100)]
    }
    
    # 创建训练器
    trainer = VillageTrainer(env, agents, 'configs/training_config.yaml')
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()