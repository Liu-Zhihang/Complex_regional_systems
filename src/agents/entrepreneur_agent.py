# Complex_regional_systems/src/agents/entrepreneur_agent.py
import numpy as np
from typing import Dict
from .base_agent import BaseAgent
from typing import Dict, Any
class EntrepreneurAgent(BaseAgent):
    """企业家智能体"""
    def __init__(self, config: Dict):
        super().__init__(config)
        # 初始化企业家特定状态
        self.state.update({
            'wealth': 50.0,
            'skill': np.random.uniform(0.5, 0.9),
            'innovation_level': 0.5,
            'market_share': 0.1,
            'employees': []
        })
        
    def act(self, state: Dict) -> Dict:
        """生成企业家动作"""
        observation = self.observe(state)
        
        # 分析当前状态
        local_resources = observation['local_view']['resources']
        market_info = observation['market_info']
        agent_state = observation['agent_state']
        
        # 决策逻辑
        investment = self._decide_investment(local_resources, market_info)
        production = self._decide_production(agent_state, market_info)
        
        # 确保返回有效的浮点数
        return {
            'investment': float(np.clip(investment, 0, 1)),
            'production': float(np.clip(production, 0, 1))
        }
    
    def _decide_investment(self, resources, market_info) -> float:
        """决定投资水平"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _decide_production(self, agent_state, market_info) -> float:
        """决定生产水平"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _decide_hiring(self, observation) -> Dict:
        """决定招聘策略"""
        return {
            'hire_count': int(self.state['wealth'] / 100),  # 每100财富雇佣1人
            'wage_level': np.clip(self.state['wealth'] / 1000, 0.1, 0.5)  # 工资水平
        }
    
    def _decide_innovation(self, observation) -> float:
        """决定创新投入"""
        # 根据市场份额和财富决定创新投入
        market_position = self.state['market_share']
        wealth_factor = self.state['wealth'] / 100.0
        
        innovation_investment = (
            0.3 * market_position + 
            0.7 * wealth_factor
        )
        return np.clip(innovation_investment, 0, 1)
    
    def _decide_movement(self, observation) -> int:
        """决定移动方向"""
        # 寻找资源丰富的区域
        local_resources = observation['local_view']['resources']
        resource_values = np.array([r['amount'] for r in local_resources.values()])
        best_direction = np.argmax(resource_values.mean(axis=0))
        return best_direction  # 0=上, 1=右, 2=下, 3=左
    
    def _update_state(self, experience: Dict):
        """更新企业家状态"""
        super()._update_state(experience)
        
        # 更新创新水平
        innovation_gain = experience.get('innovation_gain', 0)
        self.state['innovation_level'] = np.clip(
            self.state['innovation_level'] + innovation_gain,
            0, 1
        )
        
        # 更新市场份额
        market_change = experience.get('market_share_change', 0)
        self.state['market_share'] = np.clip(
            self.state['market_share'] + market_change,
            0, 1
        )
        
        # 更新员工列表
        self.state['employees'] = experience.get('employees', self.state['employees'])
    
    def _get_observation_dim(self) -> int:
        # 地形特征 (1) + 资源特征 (3) + 市场指标 (3) + 个人状态 (3)
        return 10
    
    def _get_action_dim(self) -> int:
        # investment (1) + production (1) + hiring (2) + innovation (1)
        return 5