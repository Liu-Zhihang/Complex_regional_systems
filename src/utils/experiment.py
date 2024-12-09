# Complex_regional_systems/src/utils/experiment.py
import os
import yaml
import json
import numpy as np
from typing import Dict, List
from datetime import datetime
from ..environment import VillageEnv
from ..training.trainer import VillageTrainer
from .visualization import ExperimentVisualizer

class ExperimentManager:
    """实验管理器"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.experiments = []
        self.visualizer = ExperimentVisualizer()
        
        # 创建实验目录
        self.experiment_dir = os.path.join('results', 
                                         f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def run_experiment(self, name: str, params: Dict):
        """运行单个实验"""
        print(f"Starting experiment: {name}")
        
        # 合并参数
        config = self._merge_configs(self.config, params)
        
        # 创建实验记录
        experiment = {
            'name': name,
            'params': params,
            'results': [],
            'metrics': {
                'final_reward': None,
                'convergence_episode': None,
                'training_time': None
            }
        }
        
        # 创建环境和训练器
        env = self._create_env(config)
        trainer = self._create_trainer(env, config)
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 运行训练
        for episode in range(config['training']['num_episodes']):
            metrics = trainer.train_episode()
            experiment['results'].append(metrics)
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {metrics['reward']:.2f}")
        
        # 记录训练时间
        experiment['metrics']['training_time'] = (datetime.now() - start_time).total_seconds()
        
        # 计算最终指标
        experiment['metrics']['final_reward'] = np.mean([r['reward'] for r in experiment['results'][-100:]])
        experiment['metrics']['convergence_episode'] = self._find_convergence_episode(experiment['results'])
        
        # 保存实验结果
        self._save_experiment(experiment)
        
        self.experiments.append(experiment)
        return experiment
    
    def compare_experiments(self):
        """比较多个实验结果"""
        # 生成比较图表
        self.visualizer.plot_comparison(self.experiments)
        
        # 保存比较结果
        comparison_path = os.path.join(self.experiment_dir, 'comparison.png')
        self.visualizer.save_plots(self.experiments, comparison_path)
        
        # 生成比较报告
        report = self._generate_comparison_report()
        report_path = os.path.join(self.experiment_dir, 'comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
    
    def _load_config(self, path: str) -> Dict:
        """加载配置文件"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, base_config: Dict, params: Dict) -> Dict:
        """合并配置"""
        config = base_config.copy()
        for key, value in params.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        return config
    
    def _create_env(self, config: Dict) -> VillageEnv:
        """创建环境"""
        return VillageEnv(config)
    
    def _create_trainer(self, env: VillageEnv, config: Dict) -> VillageTrainer:
        """创建训练器"""
        return VillageTrainer(env, config)
    
    def _find_convergence_episode(self, results: List[Dict]) -> int:
        """找到收敛的episode"""
        rewards = [r['reward'] for r in results]
        window = 100
        averages = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # 如果最后100个episode的标准差小于阈值，认为已收敛
        threshold = 0.1
        for i in range(len(averages)-window):
            if np.std(averages[i:i+window]) < threshold:
                return i
        return len(results)
    
    def _save_experiment(self, experiment: Dict):
        """保存实验结果"""
        # 创建实验目录
        exp_dir = os.path.join(self.experiment_dir, experiment['name'])
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(exp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(experiment['params'], f)
        
        # 保存结果
        results_path = os.path.join(exp_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(experiment['results'], f, indent=4)
        
        # 保存指标
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(experiment['metrics'], f, indent=4)
    
    def _generate_comparison_report(self) -> Dict:
        """生成比较报告"""
        report = {
            'experiments': [],
            'best_performing': None,
            'fastest_convergence': None
        }
        
        for exp in self.experiments:
            exp_summary = {
                'name': exp['name'],
                'final_reward': exp['metrics']['final_reward'],
                'convergence_episode': exp['metrics']['convergence_episode'],
                'training_time': exp['metrics']['training_time']
            }
            report['experiments'].append(exp_summary)
        
        # 找出最佳性能和最快收敛的实验
        report['best_performing'] = max(report['experiments'], 
                                      key=lambda x: x['final_reward'])['name']
        report['fastest_convergence'] = min(report['experiments'], 
                                          key=lambda x: x['convergence_episode'])['name']
        
        return report