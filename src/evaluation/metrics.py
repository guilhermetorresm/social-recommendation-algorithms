# src/evaluation/metrics.py - Versão Corrigida

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings


class Metrics:
    """Implementação corrigida de métricas para sistemas de recomendação."""
    
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
        if k <= 0 or len(recommended) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        relevant_in_k = sum(1 for item in recommended_k if item in relevant_set)
        
        return relevant_in_k / len(recommended_k)  # Use len real, não k fixo
    
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
        
        if k <= 0 or len(recommended) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        relevant_in_k = len(recommended_k & relevant_set)
        
        return relevant_in_k / len(relevant)
    
    @staticmethod
    def ndcg_at_k(recommended: List[int], relevant_scores: Dict[int, float], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@k - VERSÃO CORRIGIDA.
        
        Args:
            recommended: Lista de itens recomendados (ordenada)
            relevant_scores: Dicionário com scores de relevância (não apenas binário)
            k: Número de recomendações a considerar
            
        Returns:
            float: NDCG@k
        """
        if k <= 0 or len(recommended) == 0 or len(relevant_scores) == 0:
            return 0.0
        
        def dcg_at_k(items: List[int], scores: Dict[int, float], k: int) -> float:
            dcg = 0.0
            for i, item in enumerate(items[:k]):
                if item in scores:
                    # CORREÇÃO: Fórmula correta do DCG
                    gain = 2**scores[item] - 1
                    discount = np.log2(i + 2)  # posição i+1, log2(i+2)
                    dcg += gain / discount
            return dcg
        
        # DCG dos itens recomendados
        dcg = dcg_at_k(recommended, relevant_scores, k)
        
        # IDCG: DCG ideal (itens relevantes ordenados por score)
        ideal_items = sorted(relevant_scores.keys(), 
                           key=lambda x: relevant_scores[x], reverse=True)
        idcg = dcg_at_k(ideal_items, relevant_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def map_at_k(recommendations_dict: Dict[int, List[int]], 
                 relevant_dict: Dict[int, List[int]], k: int) -> float:
        """
        Mean Average Precision@k - VERSÃO CORRIGIDA.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            relevant_dict: Dicionário {user_id: [lista de itens relevantes]}
            k: Número de recomendações a considerar
            
        Returns:
            float: MAP@k
        """
        if k <= 0:
            return 0.0
            
        ap_scores = []
        
        for user_id, recommended in recommendations_dict.items():
            if user_id not in relevant_dict or len(relevant_dict[user_id]) == 0:
                continue
                
            relevant = relevant_dict[user_id]
            relevant_set = set(relevant)
            
            # Calcula Average Precision para este usuário
            ap = 0.0
            relevant_found = 0
            
            for i, item in enumerate(recommended[:k]):
                if item in relevant_set:
                    relevant_found += 1
                    precision_at_i = relevant_found / (i + 1)
                    ap += precision_at_i
            
            # CORREÇÃO: Normaliza pelo número de itens relevantes
            # não pelo min(len(relevant), k)
            if relevant_found > 0:
                ap /= len(relevant)  # Normaliza pelo total de relevantes
            
            ap_scores.append(ap)
        
        if len(ap_scores) == 0:
            return 0.0
        
        return np.mean(ap_scores)
    
    @staticmethod
    def coverage(recommendations_dict: Dict[int, List[int]], 
                 all_items: Set[int]) -> float:
        """
        Cobertura: proporção de itens que aparecem nas recomendações.
        """
        if len(all_items) == 0:
            return 0.0
            
        recommended_items = set()
        for items in recommendations_dict.values():
            recommended_items.update(items)
        
        return len(recommended_items) / len(all_items)
    
    @staticmethod
    def intra_list_diversity(recommendations: List[int], 
                           similarity_matrix: Optional[pd.DataFrame] = None) -> float:
        """
        Diversidade intra-lista: média da dissimilaridade entre itens recomendados.
        
        Args:
            recommendations: Lista de itens recomendados
            similarity_matrix: Matriz de similaridade entre itens (opcional)
            
        Returns:
            float: Diversidade (0 = itens idênticos, 1 = itens totalmente diferentes)
        """
        if len(recommendations) < 2:
            return 0.0
        
        if similarity_matrix is None:
            # Se não há matriz de similaridade, assume itens únicos = máxima diversidade
            return 1.0 if len(set(recommendations)) == len(recommendations) else 0.0
        
        dissimilarities = []
        n_pairs = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                
                if (item_i in similarity_matrix.index and 
                    item_j in similarity_matrix.columns):
                    similarity = similarity_matrix.loc[item_i, item_j]
                    dissimilarities.append(1 - similarity)
                    n_pairs += 1
        
        if n_pairs == 0:
            warnings.warn("Nenhum par de itens encontrado na matriz de similaridade")
            return 0.0
        
        return np.mean(dissimilarities)
    
    @staticmethod
    def novelty(recommendations_dict: Dict[int, List[int]], 
                item_popularity: Dict[int, float],
                method: str = 'log') -> float:
        """
        Novidade: o quão não-populares são os itens recomendados.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            item_popularity: Dicionário {item_id: popularidade normalizada [0,1]}
            method: Método de cálculo ('log', 'linear')
            
        Returns:
            float: Novidade média
        """
        novelty_scores = []
        
        for recommended in recommendations_dict.values():
            for item in recommended:
                if item in item_popularity:
                    popularity = item_popularity[item]
                    
                    if method == 'log':
                        # CORREÇÃO: Evita log(0) de forma mais robusta
                        novelty = -np.log2(max(popularity, 1e-8))
                    elif method == 'linear':
                        novelty = 1 - popularity
                    else:
                        raise ValueError(f"Método '{method}' não suportado")
                    
                    novelty_scores.append(novelty)
        
        if len(novelty_scores) == 0:
            return 0.0
        
        return np.mean(novelty_scores)
    
    @staticmethod
    def personalization(recommendations_dict: Dict[int, List[int]]) -> float:
        """
        Personalização: o quão diferentes são as recomendações entre usuários.
        
        Args:
            recommendations_dict: Dicionário {user_id: [lista de recomendações]}
            
        Returns:
            float: Personalização (0 = todas iguais, 1 = todas diferentes)
        """
        if len(recommendations_dict) < 2:
            return 0.0
        
        users = list(recommendations_dict.keys())
        dissimilarities = []
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                rec_i = set(recommendations_dict[users[i]])
                rec_j = set(recommendations_dict[users[j]])
                
                # Jaccard distance
                intersection = len(rec_i & rec_j)
                union = len(rec_i | rec_j)
                
                if union == 0:
                    dissimilarity = 0.0
                else:
                    dissimilarity = 1 - (intersection / union)
                
                dissimilarities.append(dissimilarity)
        
        return np.mean(dissimilarities) if dissimilarities else 0.0