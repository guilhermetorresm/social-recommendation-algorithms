# src/models/collaborative/svd.py

from surprise import SVD, Dataset, Reader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time
from surprise import KNNBasic, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from ..base_model import BaseRecommender


class SVDModel(BaseRecommender):
    """Modelo SVD (Singular Value Decomposition) usando Surprise."""
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, 
                 lr_all: float = 0.005, reg_all: float = 0.02, **kwargs):
        """
        Inicializa o modelo SVD.
        
        Args:
            n_factors: Número de fatores latentes
            n_epochs: Número de épocas de treinamento
            lr_all: Taxa de aprendizado
            reg_all: Termo de regularização
        """
        super().__init__(
            model_name="SVD",
            model_type="collaborative",
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            **kwargs
        )
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
        self.trainset = None
        self.train_data = None
        self.user_items = {}
        self.item_users = {}
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'SVDModel':
        """
        Treina o modelo SVD.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        print(f"Parâmetros: n_factors={self.n_factors}, n_epochs={self.n_epochs}")
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
                self.item_users[item_id] = set()
            self.item_users[item_id].add(user_id)
        
        # Prepara dados para Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_data[['user_id', 'item_id', 'rating']], 
            reader
        )
        self.trainset = data.build_full_trainset()
        
        # Configura e treina modelo
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        
        self.model.fit(self.trainset)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Modelo treinado em {self.training_time:.2f}s")
        print(f"Fatores latentes aprendidos para {self.trainset.n_users} usuários e {self.trainset.n_items} itens")
        
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
            # Se usuário ou item não existe, retorna média global
            return self.trainset.global_mean
    
    def get_user_factors(self, user_id: int) -> np.ndarray:
        """
        Retorna os fatores latentes de um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            np.ndarray: Vetor de fatores latentes
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            inner_id = self.trainset.to_inner_uid(user_id)
            return self.model.pu[inner_id]
        except:
            return np.zeros(self.n_factors)
    
    def get_item_factors(self, item_id: int) -> np.ndarray:
        """
        Retorna os fatores latentes de um item.
        
        Args:
            item_id: ID do item
            
        Returns:
            np.ndarray: Vetor de fatores latentes
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            inner_id = self.trainset.to_inner_iid(item_id)
            return self.model.qi[inner_id]
        except:
            return np.zeros(self.n_factors)
    
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
        
        # Gera predições para todos os candidatos
        predictions = []
        for item_id in candidate_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        # Ordena por score e retorna top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_items]
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """
        Explica uma recomendação mostrando os fatores latentes.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            Dict com explicação da recomendação
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        user_factors = self.get_user_factors(user_id)
        item_factors = self.get_item_factors(item_id)
        
        # Produto escalar dos fatores
        factor_products = user_factors * item_factors
        
        # Top fatores que mais contribuem
        top_factor_indices = np.argsort(np.abs(factor_products))[-5:][::-1]
        
        prediction = self.predict(user_id, item_id)
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': prediction,
            'global_bias': self.trainset.global_mean,
            'user_bias': self.model.bu[self.trainset.to_inner_uid(user_id)] if user_id in self.trainset._raw2inner_id_users else 0,
            'item_bias': self.model.bi[self.trainset.to_inner_iid(item_id)] if item_id in self.trainset._raw2inner_id_items else 0,
            'top_contributing_factors': [
                {
                    'factor_index': int(idx),
                    'user_value': float(user_factors[idx]),
                    'item_value': float(item_factors[idx]),
                    'contribution': float(factor_products[idx])
                }
                for idx in top_factor_indices
            ]
        }
