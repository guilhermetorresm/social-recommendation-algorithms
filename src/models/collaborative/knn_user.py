# src/models/collaborative/knn_user.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time
from surprise import KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from ..base_model import BaseRecommender


class KNNUserModel(BaseRecommender):
    """Modelo k-NN baseado em usuários usando Surprise."""
    
    def __init__(self, k: int = 20, min_k: int = 5, sim_metric: str = 'msd', **kwargs):
        """
        Inicializa o modelo k-NN user-based.
        
        Args:
            k: Número de vizinhos
            min_k: Número mínimo de vizinhos
            sim_metric: Métrica de similaridade ('cosine', 'msd', 'pearson')
        """
        super().__init__(
            model_name="KNN-User",
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
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'KNNUserModel':
        """
        Treina o modelo k-NN user-based.
        
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
        reader = Reader(rating_scale=(0.5, 5.0))  # Corrige escala conforme dataset MovieLens
        data = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        self.trainset = data.build_full_trainset()
        
        # Configura e treina modelo
        sim_options = {
            'name': self.sim_metric,
            'user_based': True,
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
        
        # Verifica se usuário e item existem no treino
        try:
            prediction = self.model.predict(user_id, item_id)
            return prediction.est
        except:
            # Retorna média global se não conseguir prever
            return self.trainset.global_mean
    
    def get_neighbors(self, user_id: int, k: int = None) -> List[Tuple[int, float]]:
        """
        Retorna os k vizinhos mais próximos usando matriz de similaridade do Surprise.
        
        Args:
            user_id: ID do usuário
            k: Número de vizinhos (None para usar o padrão)
            
        Returns:
            List[Tuple[int, float]]: Lista de (neighbor_id, similarity)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        k = k or self.k
        
        try:
            # Converte para ID interno do Surprise
            inner_id = self.trainset.to_inner_uid(user_id)
            neighbors = self.model.get_neighbors(inner_id, k=k)
            
            # Obtém matriz de similaridade (mais eficiente)
            sim_matrix = self.model.compute_similarities()
            
            # Converte de volta para IDs originais com similaridades da matriz
            result = []
            for neighbor_inner_id in neighbors:
                neighbor_id = self.trainset.to_raw_uid(neighbor_inner_id)
                similarity = sim_matrix[inner_id, neighbor_inner_id]
                result.append((neighbor_id, float(similarity)))
            
            # Ordena por similaridade decrescente
            result.sort(key=lambda x: x[1], reverse=True)
            return result
            
        except Exception as e:
            # Log do erro para debug
            print(f"Erro ao obter vizinhos para usuário {user_id}: {e}")
            return []
    
    def _calculate_similarity(self, user1: int, user2: int) -> float:
        """
        Calcula similaridade entre dois usuários.
        
        Args:
            user1: ID do primeiro usuário
            user2: ID do segundo usuário
            
        Returns:
            float: Similaridade
        """
        # Itens em comum
        items1 = set(self.user_items.get(user1, {}).keys())
        items2 = set(self.user_items.get(user2, {}).keys())
        common_items = items1 & items2
        
        if len(common_items) == 0:
            return 0.0
        
        # Calcula similaridade do cosseno
        ratings1 = np.array([self.user_items[user1][item] for item in common_items])
        ratings2 = np.array([self.user_items[user2][item] for item in common_items])
        
        if self.sim_metric == 'cosine':
            norm1 = np.linalg.norm(ratings1)
            norm2 = np.linalg.norm(ratings2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(ratings1, ratings2) / (norm1 * norm2)
        else:
            # Pearson correlation
            if len(common_items) < 2:
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
        
        # Itens já vistos pelo usuário
        seen_items = set(self.user_items.get(user_id, {}).keys())
        
        # Todos os itens possíveis
        all_items = set(self.item_users.keys())
        
        # Itens candidatos
        if exclude_seen:
            candidate_items = all_items - seen_items
        else:
            candidate_items = all_items
        
        # Gera predições para todos os candidatos
        predictions = []
        for item_id in candidate_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Ordena por score e retorna top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_items]
    
    def explain_recommendation(self, user_id: int, item_id: int, k: int = 5) -> Dict[str, Any]:
        """
        Explica por que um item foi recomendado baseado em usuários similares.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item recomendado  
            k: Número de usuários similares para explicação
            
        Returns:
            Dict com explicação da recomendação
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Obtém usuários similares
        similar_users = self.get_neighbors(user_id, k)
        
        # Encontra usuários similares que avaliaram o item
        explanation_users = []
        for similar_user_id, similarity in similar_users:
            if item_id in self.user_items.get(similar_user_id, {}):
                user_rating = self.user_items[similar_user_id][item_id]
                explanation_users.append({
                    'user_id': similar_user_id,
                    'rating': user_rating,
                    'similarity': similarity,
                    'contribution': user_rating * similarity
                })
        
        # Calcula score predito
        predicted_score = self.predict(user_id, item_id)
        
        return {
            'recommended_item': item_id,
            'predicted_score': predicted_score,
            'explanation_based_on': explanation_users[:3],  # Top 3 usuários para explicação
            'total_similar_users': len(similar_users),
            'users_who_rated_item': len(explanation_users)
        }
