# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


class Metrics:
    """Implementação de métricas para sistemas de recomendação."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Precisão@k: proporção de itens relevantes entre os k recomendados.
        
        Args:
            recommended: Lista de itens recomendados (ordenada)
            relevant: Lista de itens relevantes
            k: Número de recomendações a considerar
            
        Returns:
            float: Precisão@k
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        relevant_in_k = sum(1 for item in recommended_k if item in relevant_set)
        
        return relevant_in_k / k
    
    @staticmethod
    def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Recall@k: proporção de itens relevantes que foram recomendados.
        
        Args:
            recommended: Lista de itens recomendados (ordenada)
            relevant: Lista de itens relevantes
            k: Número de recomendações a considerar
            
        Returns:
            float: Recall@k
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        relevant_in_k = len(recommended_k & relevant_set)
        
        return relevant_in_k / len(relevant)
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant: List[int], 
                  relevance_scores: Dict[int, float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@k.
        
        Args:
            recommended: Lista de itens recomendados (ordenada)
            relevant: Lista de itens relevantes
            relevance_scores: Dicionário com scores de relevância
            k: Número de recomendações a considerar
            
        Returns:
            float: NDCG@k
        """
        def dcg_at_k(items: List[int], scores: Dict[int, float], k: int) -> float:
            dcg = 0.0
            for i, item in enumerate(items[:k]):
                if item in scores:
                    dcg += scores[item] / np.log2(i + 2)  # i+2 porque a posição começa em 1
            return dcg
        
        # DCG dos itens recomendados
        dcg = dcg_at_k(recommended, relevance_scores, k)
        
        # IDCG: DCG ideal (itens relevantes ordenados por score)
        ideal_items = sorted(relevant, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        idcg = dcg_at_k(ideal_items, relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def map_at_k(recommendations_dict: Dict[int, List[int]], 
                 relevant_dict: Dict[int, List[int]], k: int) -> float:
        """
        Mean Average Precision@k.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            relevant_dict: Dicionário {user_id: [lista de itens relevantes]}
            k: Número de recomendações a considerar
            
        Returns:
            float: MAP@k
        """
        ap_scores = []
        
        for user_id, recommended in recommendations_dict.items():
            if user_id not in relevant_dict:
                continue
                
            relevant = relevant_dict[user_id]
            if len(relevant) == 0:
                continue
            
            # Calcula Average Precision para este usuário
            relevant_set = set(relevant)
            ap = 0.0
            relevant_found = 0
            
            for i, item in enumerate(recommended[:k]):
                if item in relevant_set:
                    relevant_found += 1
                    ap += relevant_found / (i + 1)
            
            if relevant_found > 0:
                ap /= min(len(relevant), k)
            
            ap_scores.append(ap)
        
        if len(ap_scores) == 0:
            return 0.0
        
        return np.mean(ap_scores)
    
    @staticmethod
    def coverage(recommendations_dict: Dict[int, List[int]], 
                 all_items: Set[int]) -> float:
        """
        Cobertura: proporção de itens que aparecem nas recomendações.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            all_items: Conjunto de todos os itens disponíveis
            
        Returns:
            float: Cobertura
        """
        recommended_items = set()
        
        for items in recommendations_dict.values():
            recommended_items.update(items)
        
        return len(recommended_items) / len(all_items)
    
    @staticmethod
    def diversity(recommendations: List[int], 
                  similarity_matrix: pd.DataFrame) -> float:
        """
        Diversidade: média da dissimilaridade entre itens recomendados.
        
        Args:
            recommendations: Lista de itens recomendados
            similarity_matrix: Matriz de similaridade entre itens
            
        Returns:
            float: Diversidade (0 = itens idênticos, 1 = itens totalmente diferentes)
        """
        if len(recommendations) < 2:
            return 0.0
        
        dissimilarities = []
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                
                if item_i in similarity_matrix.index and item_j in similarity_matrix.columns:
                    similarity = similarity_matrix.loc[item_i, item_j]
                    dissimilarities.append(1 - similarity)
        
        if len(dissimilarities) == 0:
            return 0.0
        
        return np.mean(dissimilarities)
    
    @staticmethod
    def novelty(recommendations_dict: Dict[int, List[int]], 
                item_popularity: Dict[int, float]) -> float:
        """
        Novidade: o quão não-populares são os itens recomendados.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            item_popularity: Dicionário {item_id: popularidade}
            
        Returns:
            float: Novidade média
        """
        novelty_scores = []
        
        for recommended in recommendations_dict.values():
            for item in recommended:
                if item in item_popularity:
                    # Novidade = -log(popularidade)
                    novelty = -np.log2(item_popularity[item] + 1e-10)
                    novelty_scores.append(novelty)
        
        if len(novelty_scores) == 0:
            return 0.0
        
        return np.mean(novelty_scores)
