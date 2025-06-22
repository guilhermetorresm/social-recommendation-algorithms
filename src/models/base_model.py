# src/models/base_model.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from datetime import datetime
import os


class BaseRecommender(ABC):
    """Classe base abstrata para todos os modelos de recomendação."""
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Inicializa o modelo base.
        
        Args:
            model_name: Nome do modelo
            model_type: Tipo do modelo (baseline, collaborative, content, hybrid)
            **kwargs: Parâmetros específicos do modelo
        """
        self.model_name = model_name
        self.model_type = model_type
        self.params = kwargs
        self.is_fitted = False
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'model_type': model_type,
            'parameters': kwargs
        }
        self.training_time = None
        self.prediction_time = None
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'BaseRecommender':
        """
        Treina o modelo com os dados de treino.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            **kwargs: Parâmetros adicionais
            
        Returns:
            self: Instância do modelo treinado
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz a avaliação para um par usuário-item.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            float: Avaliação predita
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda os top-N itens para um usuário.
        
        Args:
            user_id: ID do usuário
            n_items: Número de itens a recomendar
            exclude_seen: Se deve excluir itens já vistos pelo usuário
            
        Returns:
            List[Tuple[int, float]]: Lista de tuplas (item_id, score)
        """
        pass
    
    def predict_batch(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Prediz avaliações para múltiplos pares usuário-item.
        
        Args:
            user_item_pairs: Lista de tuplas (user_id, item_id)
            
        Returns:
            np.ndarray: Array de avaliações preditas
        """
        predictions = []
        for user_id, item_id in user_item_pairs:
            pred = self.predict(user_id, item_id)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'parameters': self.params,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'metadata': self.metadata
        }
    
    def save_model(self, path: str) -> None:
        """
        Salva o modelo em disco.
        
        Args:
            path: Caminho para salvar o modelo
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_model(cls, path: str) -> 'BaseRecommender':
        """
        Carrega um modelo do disco.
        
        Args:
            path: Caminho do modelo salvo
            
        Returns:
            BaseRecommender: Modelo carregado
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return f"{self.model_name}({self.model_type})"