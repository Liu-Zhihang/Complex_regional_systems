# Complex_regional_systems/examples/basic_simulation.py
import os
import sys
import yaml
import numpy as np
from src.environment import VillageEnv
from src.agents import LeaderAgent, EntrepreneurAgent, LaborAgent

def load_config(path):
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)
    # 构建配置文件的完整路径
    config_path = os.path.join(project_root, path)
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件不存在: {config_path}")
        print(f"当前工作目录: {os.getcwd()}")
        raise

def main():
    try:
        # 加载配置
        env_config = load_config('configs/env_config.yaml')
        
        # 创建环境
        env = VillageEnv(env_config)
        
        # 创建智能体
        leader = LeaderAgent(env_config)
        entrepreneurs = [EntrepreneurAgent(env_config) for _ in range(2)]
        laborers = [LaborAgent(env_config) for _ in range(5)]
        
        # 运行模拟
        state = env.reset()
        for step in range(10):
            print(f"\nStep {step}")
            
            # 收集智能体动作
            actions = {
                "leader": leader.act(state),
                "entrepreneurs": [e.act(state) for e in entrepreneurs],
                "laborers": [l.act(state) for l in laborers]
            }
            
            # 环境步进
            state, rewards, done, info = env.step(actions)
            print(f"Rewards: {rewards}")
            
            if done:
                break
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    main()