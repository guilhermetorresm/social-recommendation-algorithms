# src/models/baseline/global_mean.py

import pandas as pd
import numpy as np
from typing import List, Tuple
import time
from ..base_model import BaseRecommender


class GlobalMeanModel(BaseRecommender):
    """Modelo baseline que prediz a média global para todos os pares usuário-item."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_name="GlobalMean",
            model_type="baseline",
            **kwargs
        )
        self.global_mean = None
        self.item_means = {}
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'GlobalMeanModel':
        """
        Treina o modelo calculando a média global.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        start_time = time.time()
        
        # Armazena dados de treino
        self.train_data = train_data.copy()
        
        # Calcula média global
        self.global_mean = train_data['rating'].mean()
        
        # Calcula média por item (para recomendações)
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Média global calculada: {self.global_mean:.3f}")
        print(f"Tempo de treinamento: {self.training_time:.2f}s")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz a avaliação (sempre retorna a média global).
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            float: Média global
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        return self.global_mean
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda itens com base na média de avaliação.
        
        Args:
            user_id: ID do usuário
            n_items: Número de itens a recomendar
            exclude_seen: Se deve excluir itens já vistos
            
        Returns:
            List[Tuple[int, float]]: Lista de tuplas (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Itens vistos pelo usuário
        seen_items = set()
        if exclude_seen:
            user_data = self.train_data[self.train_data['user_id'] == user_id]
            seen_items = set(user_data['item_id'].values)
        
        # Ordena itens por média de avaliação
        recommendations = []
        for item_id, mean_rating in self.item_means.items():
            if item_id not in seen_items:
                recommendations.append((item_id, mean_rating))
        
        # Ordena por score e retorna top-N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_items]
