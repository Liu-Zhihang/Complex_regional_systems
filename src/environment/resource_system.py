# Complex_regional_systems/src/environment/resource_system.py
import numpy as np
from typing import Dict, Tuple
from scipy.ndimage import convolve

class ResourceSystem:
    """资源系统"""
    def __init__(self, config: Dict):
        self.resource_types = config['environment']['resource_types']
        self.resource_configs = config['resources']
        self.grid_size = config['environment']['grid_size']
        
        # 扩散核
        self.diffusion_kernel = np.array([[0.05, 0.1, 0.05],
                                        [0.1, 0.4, 0.1],
                                        [0.05, 0.1, 0.05]])
        
        # 初始化资源状态
        self.resources = {
            resource_type: {
                'amount': np.zeros(self.grid_size),
                'max_capacity': self.resource_configs[resource_type]['max_capacity'],
                'regeneration_rate': self.resource_configs[resource_type]['regeneration_rate'],
                'diffusion_rate': 0.1  # 资源扩散率
            }
            for resource_type in self.resource_types
        }
        
        self._initialize_resources()
    
    def _initialize_resources(self):
        """初始化资源分布"""
        for resource_type in self.resource_types:
            max_capacity = self.resources[resource_type]['max_capacity']
            # 使用高斯分布初始化资源
            mean = max_capacity * 0.7
            std = max_capacity * 0.1
            self.resources[resource_type]['amount'] = np.clip(
                np.random.normal(mean, std, size=self.grid_size),
                0.5 * max_capacity,
                max_capacity
            )
    
    def step(self):
        """资源系统的每步更新"""
        for resource_type in self.resource_types:
            self._regenerate_resource(resource_type)
            self._diffuse_resource(resource_type)
    
    def _regenerate_resource(self, resource_type: str):
        """改进的资源再生机制"""
        resource = self.resources[resource_type]
        current_amount = resource['amount']
        max_capacity = resource['max_capacity']
        regeneration_rate = resource['regeneration_rate']
        
        # 考虑空间分布的逻辑斯蒂增长
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                current = current_amount[i,j]
                # 添加随机扰动
                noise = np.random.normal(0, 0.01)
                growth = regeneration_rate * current * (1 - current/max_capacity) + noise
                current_amount[i,j] = np.clip(current + growth, 0, max_capacity)
    
    def _diffuse_resource(self, resource_type: str):
        """资源扩散"""
        resource = self.resources[resource_type]
        current_amount = resource['amount']
        diffusion_rate = resource['diffusion_rate']
        
        # 使用卷积实现扩散
        diffused = convolve(current_amount, self.diffusion_kernel, mode='reflect')
        resource['amount'] = current_amount + diffusion_rate * (diffused - current_amount)
        
        # 确保资源量在合理范围内
        resource['amount'] = np.clip(resource['amount'], 0, resource['max_capacity'])
    
    def extract_resource(self, resource_type: str, position: Tuple[int, int], amount: float) -> float:
        """提取资源"""
        x, y = position
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return 0.0
            
        available = self.resources[resource_type]['amount'][x,y]
        extracted = min(available, amount)
        self.resources[resource_type]['amount'][x,y] -= extracted
        
        # 返回实际提取的资源量
        return extracted
    
    def get_resource_state(self) -> Dict:
        """获取资源状态"""
        return {
            resource_type: {
                'amount': resource['amount'].copy(),
                'capacity_ratio': resource['amount'] / resource['max_capacity'],
                'total_amount': np.sum(resource['amount']),
                'mean_amount': np.mean(resource['amount']),
                'std_amount': np.std(resource['amount'])
            }
            for resource_type, resource in self.resources.items()
        }
    
    def get_local_resources(self, position: Tuple[int, int], radius: int) -> Dict:
        """获取局部区域的资源状态"""
        x, y = position
        x_min = max(0, x - radius)
        x_max = min(self.grid_size[0], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.grid_size[1], y + radius + 1)
        
        return {
            resource_type: {
                'amount': resource['amount'][x_min:x_max, y_min:y_max].copy(),
                'capacity_ratio': (resource['amount'][x_min:x_max, y_min:y_max] / 
                                 resource['max_capacity'])
            }
            for resource_type, resource in self.resources.items()
        }