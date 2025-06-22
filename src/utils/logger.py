# src/utils/logger.py

import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """Sistema de logging centralizado."""
    
    _instances = {}
    
    def __new__(cls, name: str, log_dir: str = 'logs'):
        """Implementa o padrão Singleton por nome."""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        """Inicializa o logger."""
        if hasattr(self, 'initialized'):
            return
            
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove handlers existentes
        self.logger.handlers = []
        
        # Cria o diretório de logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Handler para arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}_{timestamp}.log')
        )
        file_handler.setLevel(logging.INFO)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Adiciona handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.initialized = True
    
    def info(self, message: str) -> None:
        """Log de informação."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log de aviso."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log de erro."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log de debug."""
        self.logger.debug(message)
    
    def experiment_start(self, experiment_name: str, params: dict) -> None:
        """Log do início de um experimento."""
        self.info(f"Starting experiment: {experiment_name}")
        self.info(f"Parameters: {params}")
    
    def experiment_end(self, experiment_name: str, results: dict) -> None:
        """Log do fim de um experimento."""
        self.info(f"Experiment completed: {experiment_name}")
        self.info(f"Results: {results}")
    
    def model_training_start(self, model_name: str) -> None:
        """Log do início do treinamento."""
        self.info(f"Training model: {model_name}")
    
    def model_training_end(self, model_name: str, duration: float) -> None:
        """Log do fim do treinamento."""
        self.info(f"Model {model_name} trained in {duration:.2f} seconds")
