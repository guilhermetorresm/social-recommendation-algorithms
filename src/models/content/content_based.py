import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time
from sklearn.metrics.pairwise import cosine_similarity
from ..base_model import BaseRecommender

class ContentBasedModel(BaseRecommender):
    """
    Modelo de recomendação baseado em conteúdo (gêneros dos filmes).
    Este modelo recomenda itens similares àqueles que o usuário já avaliou positivamente.
    """
    
    def __init__(self, **kwargs):
        """Inicializa o modelo Content-Based."""
        super().__init__(
            model_name="ContentBased",
            model_type="content",
            **kwargs
        )
        self.item_features = None
        self.item_similarity_matrix = None
        self.user_items = {}
        self.global_mean = 0

    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'ContentBasedModel':
        """
        Treina o modelo construindo a matriz de similaridade de itens.
        Este modelo requer um DataFrame adicional com as características dos itens.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            **kwargs: Deve conter 'items_data', um DataFrame com as features dos itens.
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        start_time = time.time()
        
        if 'items_data' not in kwargs:
            raise ValueError("O modelo ContentBased requer 'items_data' no método fit.")
        
        items_data = kwargs['items_data']
        
        # 1. Preparar features dos itens (gêneros)
        # Seleciona apenas as colunas de gênero, que são as últimas 19 no MovieLens 100K
        genre_columns = items_data.columns[-19:]
        self.item_features = items_data.set_index('item_id')[genre_columns]
        
        # 2. Calcular a matriz de similaridade de cosseno entre todos os itens
        print("Calculando a matriz de similaridade de itens...")
        cosine_sim = cosine_similarity(self.item_features)
        self.item_similarity_matrix = pd.DataFrame(
            cosine_sim, 
            index=self.item_features.index, 
            columns=self.item_features.index
        )
        
        # 3. Armazenar histórico de itens por usuário para recomendações futuras
        for _, row in train_data.iterrows():
            user_id, item_id, rating = row['user_id'], row['item_id'], row['rating']
            if user_id not in self.user_items:
                self.user_items[user_id] = []
            self.user_items[user_id].append((item_id, rating))
            
        self.global_mean = train_data['rating'].mean()
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Matriz de similaridade criada para {len(self.item_features)} itens.")
        print(f"Tempo de treinamento: {self.training_time:.2f}s")
        
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prevê a avaliação de um item para um usuário.
        A previsão é a média ponderada das avaliações do usuário para itens similares,
        onde os pesos são os scores de similaridade.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item a ser previsto
            
        Returns:
            float: Avaliação prevista
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        user_history = self.user_items.get(user_id, [])
        
        if not user_history or item_id not in self.item_similarity_matrix:
            return self.global_mean

        # Similaridades entre o item-alvo e todos os itens que o usuário avaliou
        sim_scores = self.item_similarity_matrix[item_id]
        
        total_score = 0
        total_similarity = 0
        
        for seen_item, rating in user_history:
            if seen_item in sim_scores:
                similarity = sim_scores[seen_item]
                total_score += similarity * rating
                total_similarity += similarity
        
        if total_similarity == 0:
            return self.global_mean
            
        return total_score / total_similarity

    def recommend(self, user_id: int, n_items: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
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
            
        user_history = self.user_items.get(user_id, [])
        if not user_history:
            return [] # Não é possível recomendar para um usuário sem histórico

        # Calcular scores de recomendação agregados
        # para todos os itens, baseados no histórico do usuário
        candidate_scores = pd.Series(dtype=float)
        
        # Considerar apenas itens que o usuário avaliou bem (ex: rating >= 4)
        positive_history = [item for item, rating in user_history if rating >= 4.0]
        if not positive_history:
             positive_history = [item for item, _ in user_history] # Fallback para todo o histórico

        # Agrega as similaridades dos itens que o usuário gostou
        for seen_item in positive_history:
            if seen_item in self.item_similarity_matrix:
                sim_series = self.item_similarity_matrix[seen_item]
                candidate_scores = candidate_scores.add(sim_series, fill_value=0)
        
        if candidate_scores.empty:
            return []

        # Excluir itens já vistos, se solicitado
        if exclude_seen:
            seen_items = [item for item, _ in user_history]
            candidate_scores = candidate_scores.drop(labels=seen_items, errors='ignore')
            
        # Ordenar e retornar top-N
        top_recommendations = candidate_scores.nlargest(n_items)
        
        return list(top_recommendations.items())