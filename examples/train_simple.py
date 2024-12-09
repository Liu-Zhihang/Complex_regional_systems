# Complex_regional_systems/examples/train_simple.py
import os
import sys
import yaml
import numpy as np
from src.training.trainer import VillageTrainer
from src.environment import VillageEnv
from src.agents import LeaderAgent, EntrepreneurAgent, LaborAgent
from src.utils.visualizer import VillageVisualizer

def load_config(path: str):
    """加载配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_total_reward(rewards):
    """计算总奖励，处理NaN值"""
    total = 0.0
    for agent_type, reward in rewards.items():
        if isinstance(reward, list):
            # 处理智能体列表的奖励
            valid_rewards = [r for r in reward if isinstance(r, (int, float)) and not np.isnan(r)]
            if valid_rewards:
                total += sum(valid_rewards)
        elif isinstance(reward, (int, float)) and not np.isnan(reward):
            # 处理单个智能体的奖励
            total += reward
    return total

def main():
    try:
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 加载配置
        env_config = load_config(os.path.join(project_root, 'configs/env_config.yaml'))
        train_config = load_config(os.path.join(project_root, 'configs/training_config.yaml'))
        
        # 合并配置
        config = {**env_config, **train_config}
        
        # 创建环境
        env = VillageEnv(config)
        
        # 创建智能体
        agents = {
            'leader': LeaderAgent(config),
            'entrepreneurs': [EntrepreneurAgent(config) for _ in range(2)],
            'laborers': [LaborAgent(config) for _ in range(5)]
        }
        
        # 创建训练器和可视化器
        trainer = VillageTrainer(env, agents, config)
        visualizer = VillageVisualizer(env)
        
        # 简单的交互循环
        num_episodes = 5  # 减少episode数量，先测试基本功能
        for episode in range(num_episodes):
            print(f"\nEpisode {episode}")
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 100:
                # 收集动作
                actions = {
                    'leader': agents['leader'].act(state),
                    'entrepreneurs': [e.act(state) for e in agents['entrepreneurs']],
                    'laborers': [l.act(state) for l in agents['laborers']]
                }
                
                # 环境步进
                next_state, rewards, done, _ = env.step(actions)
                
                # 计算总奖励
                episode_reward += calculate_total_reward(rewards)
                
                # 每步都渲染和保存
                visualizer.render(episode, step)
                
                state = next_state
                step += 1
            
            print(f"Episode {episode} finished with reward: {episode_reward:.2f}")
        
        # 保存动画
        visualizer.save_animation()
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    main()