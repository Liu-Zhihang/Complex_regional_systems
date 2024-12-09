# Complex_regional_systems/src/environment/space_system.py
import numpy as np
from typing import Dict, Tuple, List
import noise

class SpaceSystem:
    """空间系统：管理环境的空间结构、地形和智能体位置"""
    def __init__(self, config: Dict):
        self.grid_size = config['environment']['grid_size']
        self.resolution = config['spatial']['resolution']
        
        # 初始化地形
        self.terrain = self._generate_terrain(config['spatial']['terrain_generation'])
        
        # 初始化智能体位置
        self.agent_positions = {}
        self._initialize_agent_positions()
        
        # 邻居查找的缓存
        self.neighbor_cache = {}
        
    def _generate_terrain(self, terrain_config: Dict) -> np.ndarray:
        """使用Perlin噪声生成地形"""
        scale = terrain_config['scale']
        octaves = terrain_config['octaves']
        terrain = np.zeros(self.grid_size)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                terrain[i][j] = noise.pnoise2(
                    i/scale, 
                    j/scale, 
                    octaves=octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=self.grid_size[0],
                    repeaty=self.grid_size[1],
                    base=42
                )
        
        return (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    def _initialize_agent_positions(self):
        """初始化智能体位置"""
        # 领导者位置
        self.agent_positions['leader'] = self._get_random_valid_position()
        
        # 企业家位置
        for i in range(10):  # 10个企业家
            self.agent_positions[f'entrepreneur_{i}'] = self._get_random_valid_position()
            
        # 劳动者位置
        for i in range(100):  # 100个劳动者
            self.agent_positions[f'laborer_{i}'] = self._get_random_valid_position()
    
    def _get_random_valid_position(self) -> Tuple[int, int]:
        """获取随机有效位置"""
        while True:
            x = np.random.randint(0, self.grid_size[0])
            y = np.random.randint(0, self.grid_size[1])
            if self.terrain[x, y] > 0.2:  # 确保不在水域
                return (x, y)
    
    def get_neighbors(self, position: Tuple[int, int], radius: int) -> List[Tuple[str, Tuple[int, int]]]:
        """获取指定范围内的邻居"""
        cache_key = (position, radius)
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        
        neighbors = []
        x, y = position
        for agent_id, pos in self.agent_positions.items():
            if pos == position:
                continue
            dist = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist <= radius:
                neighbors.append((agent_id, pos))
        
        self.neighbor_cache[cache_key] = neighbors
        return neighbors
    
    def move_agent(self, agent_id: str, direction: int) -> bool:
        """移动智能体"""
        if agent_id not in self.agent_positions:
            return False
            
        current_pos = self.agent_positions[agent_id]
        x, y = current_pos
        
        # 方向: 0=上, 1=右, 2=下, 3=左
        if direction == 0:
            new_pos = (x-1, y)
        elif direction == 1:
            new_pos = (x, y+1)
        elif direction == 2:
            new_pos = (x+1, y)
        else:
            new_pos = (x, y-1)
            
        # 检查新位置是否有效
        if self._is_valid_position(new_pos):
            self.agent_positions[agent_id] = new_pos
            self.neighbor_cache = {}  # 清除缓存
            return True
        return False
    
    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        x, y = position
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False
        return self.terrain[x, y] > 0.2  # 不在水域