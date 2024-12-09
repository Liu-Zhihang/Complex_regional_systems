# Complex_regional_systems/src/utils/evaluator.py
import numpy as np
from typing import Dict, List
from .metrics import MetricsCalculator

class VillageEvaluator:
    """评估器"""
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.metrics = MetricsCalculator()
        
    def evaluate(self, num_episodes=10):
        """评估智能体性能"""
        results = []
        for _ in range(num_episodes):
            episode_metrics = self._run_episode()
            results.append(episode_metrics)
        return self._aggregate_results(results)
    
    def _run_episode(self):
        """运行一个评估episode"""
        state = self.env.reset()
        done = False
        episode_metrics = {
            'rewards': [],
            'social_welfare': [],
            'resource_sustainability': [],
            'gini_coefficient': []
        }
        
        while not done:
            # 收集动作
            actions = {
                agent_type: agent.act(state) 
                for agent_type, agent in self.agents.items()
            }
            
            # 环境步进
            next_state, rewards, done, info = self.env.step(actions)
            
            # 收集指标
            economic_metrics = self.metrics.calculate_economic_metrics(next_state)
            social_metrics = self.metrics.calculate_social_metrics(next_state)
            
            episode_metrics['rewards'].append(sum(rewards.values()))
            episode_metrics['social_welfare'].append(economic_metrics['total_wealth'])
            episode_metrics['gini_coefficient'].append(economic_metrics['gini_coefficient'])
            
            state = next_state
            
        return episode_metrics
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合多个episode的结果"""
        aggregated = {}
        for key in results[0].keys():
            values = [np.mean(episode[key]) for episode in results]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated