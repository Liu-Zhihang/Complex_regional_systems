# Complex_regional_systems/tests/test_village_env.py
import pytest
import numpy as np
import yaml
from src.environment import VillageEnv

def load_test_config():
    """加载测试配置"""
    return {
        'environment': {
            'grid_size': [10, 10],  # 使用小网格加快测试
            'max_steps': 100,
            'resource_types': ['land', 'water', 'forest']
        },
        'spatial': {
            'resolution': 30,
            'terrain_generation': {
                'algorithm': 'perlin',
                'scale': 50.0,
                'octaves': 6
            }
        },
        'resources': {
            'land': {'regeneration_rate': 0.05, 'max_capacity': 100.0},
            'water': {'regeneration_rate': 0.1, 'max_capacity': 100.0},
            'forest': {'regeneration_rate': 0.02, 'max_capacity': 100.0}
        }
    }

def get_random_action():
    """生成随机动作"""
    return {
        'leader': {
            'tax_rate': np.random.random(),
            'subsidy': np.random.random()
        },
        'entrepreneurs': [{
            'investment': np.random.random(),
            'production': np.random.random()
        } for _ in range(2)],
        'laborers': [{
            'work_effort': np.random.random(),
            'skill_learning': np.random.random()
        } for _ in range(5)]
    }

class TestVillageEnv:
    @pytest.fixture
    def env(self):
        config = load_test_config()
        return VillageEnv(config)
    
    def test_initialization(self, env):
        """测试环境初始化"""
        assert env.grid_size == [10, 10]
        assert env.max_steps == 100
        assert len(env.resource_system.resources) == 3
    
    def test_reset(self, env):
        """测试重置功能"""
        state = env.reset()
        assert 'terrain' in state
        assert 'resources' in state
        assert 'agents' in state
    
    def test_step(self, env):
        """测试步进功能"""
        env.reset()
        action = get_random_action()
        next_state, rewards, done, info = env.step(action)
        
        # 验证状态
        assert isinstance(next_state, dict)
        assert 'terrain' in next_state
        assert 'resources' in next_state
        
        # 验证奖励
        assert isinstance(rewards, dict)
        assert 'leader' in rewards
        assert 'entrepreneurs' in rewards
        assert 'laborers' in rewards
        
        # 验证done标志
        assert isinstance(done, bool)
    
    def test_resource_dynamics(self, env):
        """测试资源动态"""
        env.reset()
        initial_resources = {
            r_type: np.sum(resource['amount'])
            for r_type, resource in env.resource_system.resources.items()
        }
        
        # 执行多个步骤
        for _ in range(10):
            env.step(get_random_action())
        
        final_resources = {
            r_type: np.sum(resource['amount'])
            for r_type, resource in env.resource_system.resources.items()
        }
        
        # 验证资源变化
        for r_type in initial_resources:
            assert final_resources[r_type] != initial_resources[r_type]
    
    def test_agent_interactions(self, env):
        """测试智能体交互"""
        env.reset()
        initial_wealth = {
            'leader': env.agent_states['leader']['wealth'],
            'entrepreneurs': [e['wealth'] for e in env.agent_states['entrepreneurs']],
            'laborers': [l['wealth'] for l in env.agent_states['laborers']]
        }
        
        # 执行多个步骤
        for _ in range(10):
            env.step(get_random_action())
        
        final_wealth = {
            'leader': env.agent_states['leader']['wealth'],
            'entrepreneurs': [e['wealth'] for e in env.agent_states['entrepreneurs']],
            'laborers': [l['wealth'] for l in env.agent_states['laborers']]
        }
        
        # 验证财富变化
        assert final_wealth['leader'] != initial_wealth['leader']
        assert any(f != i for f, i in zip(final_wealth['entrepreneurs'], 
                                        initial_wealth['entrepreneurs']))
        assert any(f != i for f, i in zip(final_wealth['laborers'], 
                                        initial_wealth['laborers']))