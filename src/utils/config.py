# src/utils/config.py

import yaml
import os
from typing import Dict, Any


class Config:
    """Gerenciador de configurações do projeto."""
    
    DEFAULT_CONFIG = {
        'data': {
            'raw_path': 'data/raw',
            'processed_path': 'data/processed',
            'results_path': 'data/results',
            'default_dataset': 'movielens-100k'
        },
        'evaluation': {
            'train_test_split': 0.8,
            'random_seed': 42,
            'metrics': {
                'rating_prediction': ['rmse', 'mae'],
                'ranking': ['precision_at_k', 'recall_at_k', 'ndcg_at_k'],
                'diversity': ['coverage', 'diversity'],
                'k_values': [5, 10, 20]
            }
        },
        'models': {
            'default_params': {
                'knn': {'k': 40, 'min_k': 1},
                'svd': {'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    def __init__(self, config_path: str = None):
        """Inicializa as configurações."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._merge_configs(self.config, custom_config)
    
    def _merge_configs(self, base: Dict, custom: Dict) -> None:
        """Mescla configurações customizadas com as padrão."""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtém um valor de configuração usando notação de ponto.
        
        Args:
            key_path: Caminho da chave (ex: 'data.raw_path')
            default: Valor padrão se a chave não existir
            
        Returns:
            Valor da configuração
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Define um valor de configuração."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: str) -> None:
        """Salva as configurações em um arquivo YAML."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
