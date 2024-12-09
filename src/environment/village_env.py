# Complex_regional_systems/src/environment/village_env.py
import gym
import numpy as np
from typing import Dict, Tuple, Any, List
from .space_system import SpaceSystem
from .resource_system import ResourceSystem

class VillageEnv(gym.Env):
    """村庄环境类"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 基础参数
        self.grid_size = config['environment']['grid_size']
        self.max_steps = config['environment']['max_steps']
        self.current_step = 0
        
        # 添加resource_types属性
        self.resource_types = config['environment']['resource_types']
        
        # 初始化子系统
        self.space_system = SpaceSystem(config)
        self.resource_system = ResourceSystem(config)
        
        # 智能体状态
        self.agent_states = {
            'leader': {'wealth': 100.0, 'influence': 0.5},
            'entrepreneurs': [{'wealth': 50.0, 'skill': 0.5} for _ in range(10)],
            'laborers': [{'wealth': 10.0, 'skill': 0.3} for _ in range(100)]
        }
        
        # 定义观察和动作空间
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
    
    def _create_observation_space(self):
        """创建观察空间"""
        return gym.spaces.Dict({
            'terrain': gym.spaces.Box(low=0, high=1, shape=self.grid_size),
            'resources': gym.spaces.Dict({
                resource: gym.spaces.Box(low=0, high=1, shape=self.grid_size)
                for resource in self.config['environment']['resource_types']
            }),
            'agents': gym.spaces.Dict({
                'positions': gym.spaces.Box(low=0, high=max(self.grid_size), shape=(100, 2)),
                'states': gym.spaces.Box(low=0, high=1, shape=(100, 5))
            })
        })

    def _create_action_space(self):
        """创建动作空间"""
        return gym.spaces.Dict({
            'leader': gym.spaces.Box(low=0, high=1, shape=(2,)),  # [tax_rate, subsidy]
            'entrepreneurs': gym.spaces.Box(low=0, high=1, shape=(10, 2)),  # [investment, production]
            'laborers': gym.spaces.Box(low=0, high=1, shape=(100, 2))  # [work_effort, skill_learning]
        })
    
    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        self.space_system = SpaceSystem(self.config)
        self.resource_system = ResourceSystem(self.config)
        return self._get_observation()
    
    def step(self, actions):
        """环境步进"""
        # 打印调试信息
        print("\nEnvironment Step:")
        print(f"Action types: {list(actions.keys())}")
        
        # 执行动作
        self._execute_actions(actions)
        
        # 更新资源系统
        self.resource_system.step()
        
        # 计算奖励
        rewards = self._compute_rewards(actions)
        
        # 打印奖励信息
        print(f"Rewards: {rewards}")
        
        # 更新状态
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), rewards, done, {}
    
    def _execute_actions(self, actions):
        """执行智能体动作"""
        # 1. 领导者行动
        tax_rate = actions['leader']['tax_rate']
        subsidy = actions['leader']['subsidy']
        
        # 2. 企业家行动
        for i, entrepreneur in enumerate(actions['entrepreneurs']):
            investment = entrepreneur['investment']
            production = entrepreneur['production']
            # 计算生产效果
            self._process_production(i, investment, production)
        
        # 3. 劳动者行动
        for i, laborer in enumerate(actions['laborers']):
            work_effort = laborer['work_effort']
            skill_learning = laborer['skill_learning']
            # 更新技能和工作产出
            self._process_labor(i, work_effort, skill_learning)
    
    def _process_production(self, entrepreneur_id, investment, production):
        """处理生产过程"""
        # 获取企业家位置
        position = self.space_system.agent_positions.get(f"entrepreneur_{entrepreneur_id}")
        if position is None:
            return
        
        # 计算资源消耗和产出
        for resource_type in self.config['environment']['resource_types']:
            consumed = self.resource_system.extract_resource(
                resource_type, 
                position, 
                production * investment
            )
            # 更新企业家财富
            self.agent_states['entrepreneurs'][entrepreneur_id]['wealth'] += consumed * 1.5
    
    def _process_labor(self, laborer_id, work_effort, skill_learning):
        """处理劳动过程"""
        # 更新技能
        current_skill = self.agent_states['laborers'][laborer_id]['skill']
        skill_increase = skill_learning * 0.1 * (1 - current_skill)
        self.agent_states['laborers'][laborer_id]['skill'] += skill_increase
        
        # 计算工作产出
        output = work_effort * current_skill
        self.agent_states['laborers'][laborer_id]['wealth'] += output
    
    def _get_observation(self):
        """获取当前状态观察"""
        # 添加市场信息
        market_info = {
            'prices': {'resource_' + str(i): np.random.random() for i in range(3)},
            'trade_volume': {'resource_' + str(i): np.random.random() for i in range(3)},
            'demand': {'resource_' + str(i): np.random.random() for i in range(3)}
        }
        
        # 添加社会信息
        social_info = {
            'gini_coefficient': self._calculate_gini([a['wealth'] for a in self.agent_states['laborers']]),
            'poverty_rate': np.random.random(),
            'social_network': {}
        }
        
        return {
            'terrain': self.space_system.terrain,
            'resources': self.resource_system.get_resource_state(),
            'agents': {
                'positions': np.array(list(self.space_system.agent_positions.values())),
                'states': np.array([agent['wealth'] for agent in self.agent_states['laborers']])
            },
            'market_info': market_info,
            'social_info': social_info
        }
    
    def _compute_rewards(self, actions):
        """计算奖励"""
        try:
            # 计算领导者奖励
            leader_reward = 0.0
            if 'leader' in actions:
                tax_rate = float(actions['leader'].get('tax_rate', 0.0))
                subsidy = float(actions['leader'].get('subsidy', 0.0))
                leader_reward = 0.5 * (1.0 - abs(tax_rate - 0.3)) + 0.5 * (1.0 - abs(subsidy - 0.2))
            
            # 计算企业家奖励
            entrepreneur_rewards = []
            if 'entrepreneurs' in actions:
                for e_action in actions['entrepreneurs']:
                    investment = float(e_action.get('investment', 0.0))
                    production = float(e_action.get('production', 0.0))
                    e_reward = 0.6 * production + 0.4 * investment
                    entrepreneur_rewards.append(e_reward)
            
            # 计算劳动者奖励
            laborer_rewards = []
            if 'laborers' in actions:
                for l_action in actions['laborers']:
                    work_effort = float(l_action.get('work_effort', 0.0))
                    skill_learning = float(l_action.get('skill_learning', 0.0))
                    l_reward = 0.7 * work_effort + 0.3 * skill_learning
                    laborer_rewards.append(l_reward)
            
            rewards = {
                'leader': leader_reward,
                'entrepreneurs': entrepreneur_rewards,
                'laborers': laborer_rewards
            }
            
            # 打印调试信息
            print("\nReward Computation:")
            print(f"Leader reward: {leader_reward}")
            print(f"Entrepreneur rewards: {entrepreneur_rewards}")
            print(f"Laborer rewards: {laborer_rewards}")
            
            return rewards
            
        except Exception as e:
            print(f"Error in reward computation: {str(e)}")
            # 返回默认奖励
            return {
                'leader': 0.0,
                'entrepreneurs': [0.0] * len(actions['entrepreneurs']),
                'laborers': [0.0] * len(actions['laborers'])
            }
    
    def _compute_leader_reward(self, action):
        """计算领导者奖励"""
        try:
            tax_rate = float(action.get('tax_rate', 0.0))
            subsidy = float(action.get('subsidy', 0.0))
            
            # 简化的奖励计算
            tax_efficiency = 1.0 - abs(tax_rate - 0.3)  # 最优税率假设为0.3
            subsidy_efficiency = 1.0 - abs(subsidy - 0.2)  # 最优补贴率假设为0.2
            
            return 0.5 * tax_efficiency + 0.5 * subsidy_efficiency
            
        except Exception as e:
            print(f"Error in leader reward computation: {str(e)}")
            return 0.0
    
    def _compute_entrepreneur_reward(self, e_id, action):
        """计算企业家奖励"""
        try:
            investment = float(action.get('investment', 0.0))
            production = float(action.get('production', 0.0))
            
            # 简化的奖励计算
            return 0.6 * production + 0.4 * investment
            
        except Exception as e:
            print(f"Error in entrepreneur reward computation: {str(e)}")
            return 0.0
    
    def _compute_laborer_reward(self, l_id, action):
        """计算劳动者奖励"""
        try:
            work_effort = float(action.get('work_effort', 0.0))
            skill_learning = float(action.get('skill_learning', 0.0))
            
            # 简化的奖励计算
            return 0.7 * work_effort + 0.3 * skill_learning
            
        except Exception as e:
            print(f"Error in laborer reward computation: {str(e)}")
            return 0.0
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        values = np.array(values)
        if np.all(values == 0):
            return 0
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * values).sum() / (n * values.sum())