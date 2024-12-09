# Complex_regional_systems/src/agents/leader_agent.py
import numpy as np
from typing import Dict, Any
from .base_agent import BaseAgent

class LeaderAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state['influence'] = 1.0
        self.state['wealth'] = 100.0
        
    def act(self, state: Dict) -> Dict:
        """生成领导者动作"""
        observation = self.observe(state)
        
        # 分析当前状态
        resource_state = observation['local_view']['resources']
        market_info = observation['market_info']
        social_info = observation['social_info']
        
        # 决策逻辑
        tax_rate = self._determine_tax_rate(market_info, social_info)
        subsidy = self._determine_subsidy(resource_state, social_info)
        
        # 确保返回有效的浮点数
        return {
            'tax_rate': float(np.clip(tax_rate, 0, 1)),
            'subsidy': float(np.clip(subsidy, 0, 1))
        }
    
    def _get_observation_dim(self) -> int:
        """获取观察空间维度"""
        # 地形特征 (1) + 资源特征 (3) + 社会指标 (2) + 经济指标 (2)
        return 8
    
    def _get_action_dim(self) -> int:
        """获取动作空间维度"""
        # tax_rate (1) + subsidy (1) + regulations (3)
        return 5
    
    def _determine_tax_rate(self, market_info: Dict, social_info: Dict) -> float:
        """确定税率"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _determine_subsidy(self, resource_state: Dict, social_info: Dict) -> float:
        """确定补贴"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _determine_regulations(self, observation: Dict) -> Dict:
        """确定监管政策"""
        return {
            'resource_protection': 0.5,
            'market_regulation': 0.3,
            'social_protection': 0.4
        }