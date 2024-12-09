# Complex_regional_systems/src/agents/base_agent.py
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from ..models.policy_network import PolicyNetwork, ValueNetwork
from ..models.losses import PolicyGradientLoss, ValueLoss

class BaseAgent(nn.Module):
    """基础智能体类"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.policy_net = PolicyNetwork(
            input_dim=self._get_observation_dim(),
            output_dim=self._get_action_dim(),
            hidden_sizes=config['networks']['policy_net']['hidden_sizes']
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            input_dim=self._get_observation_dim(),
            hidden_sizes=config['networks']['value_net']['hidden_sizes']
        ).to(self.device)
        
        # 初始化损失函数
        self.policy_loss = PolicyGradientLoss()
        self.value_loss = ValueLoss()
        
        # 初始化优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config['networks']['policy_net']['learning_rate']
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=config['networks']['value_net']['learning_rate']
        )
        
        self.state = {
            'wealth': 0.0,
            'skill': 0.5,
            'experience': 0.0,
            'social_capital': 0.0,
            'influence': 0.0
        }
        self.history = []
    
    def parameters(self):
        """返回所有需要优化的参数"""
        return list(self.policy_net.parameters()) + list(self.value_net.parameters())
    
    def forward(self, x):
        """前向传播"""
        policy_output = self.policy_net(x)
        value_output = self.value_net(x)
        return policy_output, value_output
    
    def observe(self, state: Dict) -> Dict:
        """观察环境状态"""
        local_view = self._get_local_view(state)
        social_info = self._get_social_info(state)
        market_info = self._get_market_info(state)
        
        return {
            'local_view': local_view,
            'social_info': social_info,
            'market_info': market_info,
            'agent_state': self.state
        }
    
    def act(self, state: Dict) -> Dict:
        """根据观察做出决策"""
        raise NotImplementedError
        
    def learn(self, experience: Dict):
        """从经验中学习"""
        self.history.append(experience)
        self._update_state(experience)
    
    def _get_local_view(self, state: Dict) -> Dict:
        """获取局部视图"""
        return {
            'terrain': state['terrain'],
            'resources': state['resources'],
            'nearby_agents': state['agents']
        }
    
    def _get_social_info(self, state: Dict) -> Dict:
        """获取社会信息"""
        return {
            'social_network': state.get('social_network', {}),
            'cooperation_opportunities': state.get('cooperation_opportunities', [])
        }
    
    def _get_market_info(self, state: Dict) -> Dict:
        """获取市场信息"""
        return {
            'prices': state.get('prices', {}),
            'trade_volume': state.get('trade_volume', {}),
            'demand': state.get('demand', {})
        }
    
    def _update_state(self, experience: Dict):
        """更新智能体状态"""
        # 更新财富
        self.state['wealth'] += experience.get('reward', 0)
        
        # 更新技能
        skill_increase = experience.get('skill_gain', 0)
        self.state['skill'] = min(1.0, self.state['skill'] + skill_increase)
        
        # 更新经验
        self.state['experience'] += 0.1
        
        # 更新社会资本
        social_gain = experience.get('social_gain', 0)
        self.state['social_capital'] += social_gain
    
    def compute_loss(self, batch):
        """计算智能体的损失"""
        try:
            # 将状态转换为张量
            states = torch.FloatTensor([
                self._preprocess_state(t[0]) for t in batch
            ]).to(self.device)
            
            # 将动作转换为张量
            actions = torch.FloatTensor([
                self._preprocess_action(t[1]) for t in batch
            ]).to(self.device)
            
            # 将奖励转换为张量
            rewards = torch.FloatTensor([
                float(t[2]) if isinstance(t[2], (int, float)) else float(t[2]['reward'])
                for t in batch
            ]).to(self.device)
            
            # 将下一个状态转换为张量
            next_states = torch.FloatTensor([
                self._preprocess_state(t[3]) for t in batch
            ]).to(self.device)
            
            # 将done标志转换为张量
            dones = torch.FloatTensor([float(t[4]) for t in batch]).to(self.device)
            
            # 计算策略损失
            action_mean, action_std = self.policy_net(states)
            policy_loss = self.policy_loss(action_mean, action_std, actions, rewards)
            
            # 计算价值损失
            values = self.value_net(states)
            next_values = self.value_net(next_states)
            target_values = rewards + self.config['training']['gamma'] * next_values * (1 - dones)
            value_loss = self.value_loss(values, target_values.detach())
            
            # 总损失
            total_loss = (policy_loss * self.config['losses']['policy_loss_weight'] + 
                         value_loss * self.config['losses']['value_loss_weight'])
            
            return total_loss
            
        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _preprocess_state(self, state: Dict) -> np.ndarray:
        """预处理状态数据"""
        # 将字典状态转换为向量
        state_vector = []
        
        # 添加地形信息
        if 'terrain' in state:
            state_vector.append(np.mean(state['terrain']))
        
        # 添加资源信息
        if 'resources' in state:
            for resource in state['resources'].values():
                state_vector.append(np.mean(resource['amount']))
        
        # 添加智能体状态
        if 'agents' in state:
            if 'states' in state['agents']:
                state_vector.extend(state['agents']['states'].flatten())
        
        return np.array(state_vector, dtype=np.float32)

    def _preprocess_action(self, action: Dict) -> np.ndarray:
        """预处理动作数据"""
        # 将字典动作转换为向量
        action_vector = []
        
        # 根据智能体类型处理不同的动作
        if isinstance(action, dict):
            for key, value in action.items():
                if isinstance(value, (int, float)):
                    action_vector.append(float(value))
                elif isinstance(value, dict):
                    action_vector.extend(value.values())
        
        return np.array(action_vector, dtype=np.float32)
    
    def _get_observation_dim(self):
        """获取观察空间维度"""
        raise NotImplementedError
        
    def _get_action_dim(self):
        """获取动作空间维度"""
        raise NotImplementedError

# Complex_regional_systems/src/agents/leader_agent.py
from .base_agent import BaseAgent

class LeaderAgent(BaseAgent):
    def act(self, state):
        return {
            "tax_rate": np.random.random(),  # 0-1之间的税率
            "subsidy": np.random.random()    # 0-1之间的补贴
        }

# Complex_regional_systems/src/agents/entrepreneur_agent.py
from .base_agent import BaseAgent

class EntrepreneurAgent(BaseAgent):
    def act(self, state):
        return {
            "investment": np.random.random(),  # 0-1之间的投资
            "production": np.random.random()   # 0-1之间的生产强度
        }

# Complex_regional_systems/src/agents/labor_agent.py
from .base_agent import BaseAgent

class LaborAgent(BaseAgent):
    def act(self, state):
        return {
            "work_effort": np.random.random(),  # 0-1之间的工作努力
            "skill_learning": np.random.random() # 0-1之间的学习投入
        }