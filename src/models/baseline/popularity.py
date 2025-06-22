# src/models/baseline/popularity.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import time
from ..base_model import BaseRecommender


class PopularityModel(BaseRecommender):
    """Modelo baseline que recomenda itens mais populares."""
    
    def __init__(self, popularity_metric: str = 'rating_count', **kwargs):
        """
        Inicializa o modelo de popularidade.
        
        Args:
            popularity_metric: Métrica de popularidade ('rating_count' ou 'mean_rating')
        """
        super().__init__(
            model_name="Popularity",
            model_type="baseline",
            popularity_metric=popularity_metric,
            **kwargs
        )
        self.popularity_metric = popularity_metric
        self.item_popularity = {}
        self.item_stats = {}
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'PopularityModel':
        """
        Treina o modelo calculando a popularidade dos itens.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name} com métrica: {self.popularity_metric}...")
        start_time = time.time()
        
        # Armazena dados de treino
        self.train_data = train_data.copy()
        
        # Calcula estatísticas por item
        item_stats = train_data.groupby('item_id').agg({
            'rating': ['count', 'mean', 'sum']
        })
        item_stats.columns = ['rating_count', 'mean_rating', 'sum_rating']
        self.item_stats = item_stats.to_dict('index')
        
        # Define popularidade baseada na métrica escolhida
        if self.popularity_metric == 'rating_count':
            self.item_popularity = {
                item_id: stats['rating_count'] 
                for item_id, stats in self.item_stats.items()
            }
        elif self.popularity_metric == 'mean_rating':
            # Combina média com número mínimo de avaliações
            min_ratings = 5
            self.item_popularity = {}
            for item_id, stats in self.item_stats.items():
                if stats['rating_count'] >= min_ratings:
                    self.item_popularity[item_id] = stats['mean_rating']
                else:
                    # Penaliza itens com poucas avaliações
                    self.item_popularity[item_id] = stats['mean_rating'] * (
                        stats['rating_count'] / min_ratings
                    )
        else:
            raise ValueError(f"Métrica desconhecida: {self.popularity_metric}")
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Popularidade calculada para {len(self.item_popularity)} itens")
        print(f"Tempo de treinamento: {self.training_time:.2f}s")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz a avaliação com base na popularidade.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            float: Score de popularidade normalizado
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Retorna a média do item se disponível
        if item_id in self.item_stats:
            return self.item_stats[item_id]['mean_rating']
        else:
            # Retorna a média global
            return self.train_data['rating'].mean()
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda os itens mais populares.
        
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
        
        # Filtra e ordena por popularidade
        recommendations = []
        for item_id, popularity in self.item_popularity.items():
            if item_id not in seen_items:
                recommendations.append((item_id, popularity))
        
        # Ordena por popularidade
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_items]
    
    def get_most_popular_items(self, n: int = 10) -> List[Tuple[int, Dict]]:
        """
        Retorna os N itens mais populares com suas estatísticas.
        
        Args:
            n: Número de itens
            
        Returns:
            List[Tuple[int, Dict]]: Lista de (item_id, estatísticas)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Ordena por popularidade
        sorted_items = sorted(
            self.item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        # Retorna com estatísticas completas
        result = []
        for item_id, popularity in sorted_items:
            stats = self.item_stats[item_id].copy()
            stats['popularity_score'] = popularity
            result.append((item_id, stats))
        
        return result
