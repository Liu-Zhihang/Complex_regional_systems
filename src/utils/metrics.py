# Complex_regional_systems/src/utils/metrics.py
import numpy as np
from typing import Dict, List

class MetricsCalculator:
    """指标计算器"""
    def __init__(self):
        self.history = {
            'rewards': [],
            'social_welfare': [],
            'gini_coefficient': [],
            'resource_sustainability': []
        }
    
    def calculate_economic_metrics(self, state: Dict) -> Dict:
        """计算经济指标"""
        total_wealth = sum(
            agent['wealth'] for agents in state['agent_states'].values()
            for agent in (agents if isinstance(agents, list) else [agents])
        )
        
        wealth_distribution = [
            agent['wealth'] for agents in state['agent_states'].values()
            for agent in (agents if isinstance(agents, list) else [agents])
        ]
        
        gini = self._calculate_gini(wealth_distribution)
        
        return {
            'total_wealth': total_wealth,
            'gini_coefficient': gini
        }
    
    def calculate_social_metrics(self, state: Dict) -> Dict:
        """计算社会指标"""
        skill_levels = [
            agent['skill'] for agents in state['agent_states'].values()
            for agent in (agents if isinstance(agents, list) else [agents])
            if 'skill' in agent
        ]
        
        return {
            'avg_skill': np.mean(skill_levels),
            'skill_std': np.std(skill_levels)
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        values = np.array(values)
        if np.all(values == 0):
            return 0
        values = values.flatten()
        n = len(values)
        indices = np.argsort(values)
        values = values[indices]
        weights = np.arange(1, n + 1) / n
        return ((2 * weights - 1) * values).sum() / (n * values.mean())
    
    def update(self, metrics: Dict):
        """更新历史指标"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_summary(self) -> Dict:
        """获取指标摘要"""
        return {
            key: {
                'mean': np.mean(values[-100:]),
                'std': np.std(values[-100:])
            }
            for key, values in self.history.items()
        }