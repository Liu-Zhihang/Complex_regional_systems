# Complex_regional_systems/src/agents/labor_agent.py
import numpy as np
from .base_agent import BaseAgent
from typing import Dict, Any
class LaborAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state['skill'] = np.random.uniform(0.2, 0.8)
        self.state['wealth'] = 10.0
        
    def act(self, state: Dict) -> Dict:
        """生成劳动者动作"""
        observation = self.observe(state)
        
        # 分析当前状态
        local_resources = observation['local_view']['resources']
        market_info = observation['market_info']
        agent_state = observation['agent_state']
        
        # 决策逻辑
        work_effort = self._decide_work_effort(local_resources, market_info)
        skill_learning = self._decide_skill_learning(agent_state)
        
        # 确保返回有效的浮点数
        return {
            'work_effort': float(np.clip(work_effort, 0, 1)),
            'skill_learning': float(np.clip(skill_learning, 0, 1))
        }
    
    def _decide_work_effort(self, resources, market_info) -> float:
        """决定工作努力程度"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _decide_skill_learning(self, agent_state) -> float:
        """决定技能学习投入"""
        # 使用简单的随机策略，后续可以改进为基于状态的决策
        return np.random.random()
    
    def _decide_movement(self, observation) -> int:
        """决定移动方向"""
        # 简单的随机移动策略
        return np.random.randint(0, 4)  # 0=上, 1=右, 2=下, 3=左
    
    def _get_observation_dim(self) -> int:
        # 地形特征 (1) + 资源特征 (3) + 市场指标 (2) + 个人状态 (2)
        return 8
    
    def _get_action_dim(self) -> int:
        # work_effort (1) + skill_learning (1) + movement (1)
        return 3