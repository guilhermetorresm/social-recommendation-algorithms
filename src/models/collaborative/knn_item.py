# src/models/collaborative/knn_item.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import time
from surprise import KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from ..base_model import BaseRecommender


class KNNItemModel(BaseRecommender):
    """Modelo k-NN baseado em itens usando Surprise."""
    
    def __init__(self, k: int = 40, min_k: int = 1, sim_metric: str = 'cosine', **kwargs):
        """
        Inicializa o modelo k-NN item-based.
        
        Args:
            k: Número de vizinhos
            min_k: Número mínimo de vizinhos
            sim_metric: Métrica de similaridade
        """
        super().__init__(
            model_name="KNN-Item",
            model_type="collaborative",
            k=k,
            min_k=min_k,
            sim_metric=sim_metric,
            **kwargs
        )
        self.k = k
        self.min_k = min_k
        self.sim_metric = sim_metric
        self.model = None
        self.trainset = None
        self.train_data = None
        self.user_items = {}
        self.item_users = {}
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'KNNItemModel':
        """
        Treina o modelo k-NN item-based.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        start_time = time.time()
        
        # Armazena dados de treino
        self.train_data = train_data.copy()
        
        # Cria estruturas auxiliares
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            if user_id not in self.user_items:
                self.user_items[user_id] = {}
            self.user_items[user_id][item_id] = rating
            
            if item_id not in self.item_users:
                self.item_users[item_id] = {}
            self.item_users[item_id][user_id] = rating
        
        # Prepara dados para Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        self.trainset = data.build_full_trainset()
        
        # Configura e treina modelo
        sim_options = {
            'name': self.sim_metric,
            'user_based': False,  # Item-based
            'min_support': self.min_k
        }
        
        self.model = KNNBasic(
            k=self.k,
            min_k=self.min_k,
            sim_options=sim_options
        )
        
        self.model.fit(self.trainset)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Modelo treinado com k={self.k}, métrica={self.sim_metric}")
        print(f"Tempo de treinamento: {self.training_time:.2f}s")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz a avaliação para um par usuário-item.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            float: Avaliação predita
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            prediction = self.model.predict(user_id, item_id)
            return prediction.est
        except:
            return self.trainset.global_mean
    
    def get_similar_items(self, item_id: int, k: int = None) -> List[Tuple[int, float]]:
        """
        Retorna os k itens mais similares.
        
        Args:
            item_id: ID do item
            k: Número de itens similares
            
        Returns:
            List[Tuple[int, float]]: Lista de (item_id, similarity)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        k = k or self.k
        
        try:
            # Converte para ID interno
            inner_id = self.trainset.to_inner_iid(item_id)
            neighbors = self.model.get_neighbors(inner_id, k=k)
            
            # Converte de volta
            result = []
            for neighbor_inner_id in neighbors:
                neighbor_id = self.trainset.to_raw_iid(neighbor_inner_id)
                # Similaridade simplificada
                similarity = self._calculate_item_similarity(item_id, neighbor_id)
                result.append((neighbor_id, similarity))
            
            return result
        except:
            return []
    
    def _calculate_item_similarity(self, item1: int, item2: int) -> float:
        """Calcula similaridade entre dois itens."""
        users1 = set(self.item_users.get(item1, {}).keys())
        users2 = set(self.item_users.get(item2, {}).keys())
        common_users = users1 & users2
        
        if len(common_users) == 0:
            return 0.0
        
        ratings1 = np.array([self.item_users[item1][user] for user in common_users])
        ratings2 = np.array([self.item_users[item2][user] for user in common_users])
        
        if self.sim_metric == 'cosine':
            norm1 = np.linalg.norm(ratings1)
            norm2 = np.linalg.norm(ratings2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(ratings1, ratings2) / (norm1 * norm2)
        else:
            if len(common_users) < 2:
                return 0.0
            return np.corrcoef(ratings1, ratings2)[0, 1]
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda os top-N itens para um usuário.
        
        Args:
            user_id: ID do usuário
            n_items: Número de itens a recomendar
            exclude_seen: Se deve excluir itens já vistos
            
        Returns:
            List[Tuple[int, float]]: Lista de tuplas (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Itens já vistos
        seen_items = set(self.user_items.get(user_id, {}).keys())
        
        # Todos os itens
        all_items = set(self.item_users.keys())
        
        # Candidatos
        if exclude_seen:
            candidate_items = all_items - seen_items
        else:
            candidate_items = all_items
        
        # Predições
        predictions = []
        for item_id in candidate_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Ordena e retorna top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_items]

