# src/models/hybrid/hybrid_model.py

import pandas as pd
import numpy as np
from typing import List, Tuple
import time

from ..base_model import BaseRecommender

class HybridModel(BaseRecommender):
    """
    Modelo Híbrido que combina as predições de dois outros modelos
    usando uma média ponderada. Esta versão é robusta e autossuficiente.
    """
    
    def __init__(self, model1: BaseRecommender, model2: BaseRecommender, weight1: float = 0.5, **kwargs):
        """
        Inicializa o Modelo Híbrido.
        """
        model_name = f"Hybrid_{model1.model_name}_{model2.model_name}_w{weight1}"
        super().__init__(model_name=model_name, model_type="hybrid", **kwargs)
        
        if not (0 <= weight1 <= 1):
            raise ValueError("O peso 'weight1' deve estar entre 0 e 1.")
            
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = 1 - weight1
        
        # ATRIBUTOS NOVOS/MODIFICADOS para tornar o modelo autossuficiente
        self.user_items = {}
        self._all_item_ids = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'HybridModel':
        """
        Treina os modelos subjacentes e constrói seu próprio histórico de interações.
        """
        print(f"Treinando {self.model_name}...")
        start_time = time.time()
        
        # 1. Treinar sub-modelos (com a lógica corrigida para kwargs)
        print(f"Treinando sub-modelo 1: {self.model1.model_name}")
        fit_kwargs1 = {}
        if self.model1.model_type == 'content' and 'items_data' in kwargs:
            fit_kwargs1['items_data'] = kwargs['items_data']
        self.model1.fit(train_data, **fit_kwargs1)
        
        print(f"Treinando sub-modelo 2: {self.model2.model_name}")
        fit_kwargs2 = {}
        if self.model2.model_type == 'content' and 'items_data' in kwargs:
            fit_kwargs2['items_data'] = kwargs['items_data']
        self.model2.fit(train_data, **fit_kwargs2)
        
        # 2. Construir o próprio histórico de usuário do modelo híbrido
        print("Construindo histórico de usuários para o modelo híbrido...")
        for _, row in train_data.iterrows():
            user_id, item_id, rating = int(row['user_id']), int(row['item_id']), row['rating']
            if user_id not in self.user_items:
                self.user_items[user_id] = []
            self.user_items[user_id].append((item_id, rating))
            
        self._all_item_ids = train_data['item_id'].unique()
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Tempo de treinamento total do modelo híbrido: {self.training_time:.2f}s")
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prevê uma avaliação combinando as previsões dos dois modelos.
        """
        pred1 = self.model1.predict(user_id, item_id)
        pred2 = self.model2.predict(user_id, item_id)
        
        hybrid_prediction = (self.weight1 * pred1) + (self.weight2 * pred2)
        return hybrid_prediction

    def predict_batch(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Gera predições para um batch de pares (usuário, item) para otimização.
        """
        # Delega a predição em lote para os sub-modelos.
        # É importante que os sub-modelos também implementem predict_batch.
        # Se não, um loop simples seria um fallback necessário no BaseRecommender.
        predictions1 = self.model1.predict_batch(user_item_pairs)
        predictions2 = self.model2.predict_batch(user_item_pairs)
        
        hybrid_predictions = (self.weight1 * predictions1) + (self.weight2 * predictions2)
        return hybrid_predictions

    def recommend(self, user_id: int, n_items: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Gera recomendações usando o histórico de usuários próprio e scores híbridos.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")

        # Usa seu próprio dicionário user_items para obter os itens já vistos
        seen_items = []
        if exclude_seen:
            seen_items = [item_id for item_id, _ in self.user_items.get(user_id, [])]
        
        candidate_items = [item_id for item_id in self._all_item_ids if item_id not in seen_items]
        
        if not candidate_items:
            return []

        # Usa predict_batch para calcular scores de forma eficiente
        user_item_pairs = [(user_id, item_id) for item_id in candidate_items]
        scores = self.predict_batch(user_item_pairs)
        
        recommendations = list(zip(candidate_items, scores))
        
        # Ordena pela pontuação e retorna o top-N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_items]