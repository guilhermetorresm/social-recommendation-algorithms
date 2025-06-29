# src/models/collaborative/svd.py

from surprise import SVD, Dataset, Reader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import warnings
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
        
        # Cria estruturas auxiliares otimizadas
        self.user_items = train_data.groupby('user_id').apply(
            lambda x: dict(zip(x['item_id'], x['rating']))
        ).to_dict()
        
        self.item_users = train_data.groupby('item_id')['user_id'].apply(set).to_dict()
        
        # Prepara dados para Surprise
        reader = Reader(rating_scale=(0.5, 5.0))  # Corrige escala conforme dataset MovieLens
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
        except Exception as e:
            # Log do erro para debug
            warnings.warn(f"Erro ao predizer para usuário {user_id}, item {item_id}: {e}")
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
            return self.model.pu[inner_id].copy()  # Cópia para evitar modificações acidentais
        except Exception as e:
            warnings.warn(f"Erro ao obter fatores do usuário {user_id}: {e}")
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
            return self.model.qi[inner_id].copy()  # Cópia para evitar modificações acidentais
        except Exception as e:
            warnings.warn(f"Erro ao obter fatores do item {item_id}: {e}")
            return np.zeros(self.n_factors)
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda os top-N itens para um usuário usando operações vetorizadas.
        
        Args:
            user_id: ID do usuário
            n_items: Número de itens a recomendar
            exclude_seen: Se deve excluir itens já vistos
            
        Returns:
            List[Tuple[int, float]]: Lista de tuplas (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            # Obter fatores do usuário
            user_factors = self.get_user_factors(user_id)
            
            # Se não conseguir obter fatores, usar método tradicional
            if np.allclose(user_factors, 0):
                return self._recommend_fallback(user_id, n_items, exclude_seen)
            
            # Itens já vistos
            seen_items = set(self.user_items.get(user_id, {}).keys())
            
            # Todos os itens
            all_items = set(self.item_users.keys())
            
            # Candidatos
            if exclude_seen:
                candidate_items = list(all_items - seen_items)
            else:
                candidate_items = list(all_items)
            
            if not candidate_items:
                return []
            
            # Predições vetorizadas para candidatos
            predictions = []
            for item_id in candidate_items:
                pred = self.predict(user_id, item_id)
                predictions.append((item_id, pred))
            
            # Ordena por score e retorna top-N
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions[:n_items]
            
        except Exception as e:
            warnings.warn(f"Erro ao gerar recomendações para usuário {user_id}: {e}")
            return []
    
    def _recommend_fallback(self, user_id: int, n_items: int = 10, 
                           exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Método de fallback para recomendação quando fatores não estão disponíveis."""
        seen_items = set(self.user_items.get(user_id, {}).keys())
        all_items = set(self.item_users.keys())
        
        if exclude_seen:
            candidate_items = all_items - seen_items
        else:
            candidate_items = all_items
        
        # Retorna itens mais populares como fallback
        item_popularity = [(item_id, len(self.item_users.get(item_id, set())))
                          for item_id in candidate_items]
        item_popularity.sort(key=lambda x: x[1], reverse=True)
        
        return [(item_id, float(popularity)) for item_id, popularity in item_popularity[:n_items]]
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """
        Explica uma recomendação mostrando os fatores latentes e componentes do SVD.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            Dict com explicação detalhada da recomendação
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            user_factors = self.get_user_factors(user_id)
            item_factors = self.get_item_factors(item_id)
            
            # Verifica se usuário/item existem
            user_exists = user_id in self.trainset._raw2inner_id_users
            item_exists = item_id in self.trainset._raw2inner_id_items
            
            if not user_exists and not item_exists:
                return {
                    'error': f'Usuário {user_id} e item {item_id} não existem no modelo',
                    'fallback_prediction': self.trainset.global_mean
                }
            
            # Produto escalar dos fatores
            factor_products = user_factors * item_factors
            
            # Top fatores que mais contribuem (positivos e negativos)
            factor_abs = np.abs(factor_products)
            top_factor_indices = np.argsort(factor_abs)[-5:][::-1]
            
            prediction = self.predict(user_id, item_id)
            
            # Componentes da predição SVD: predição = global_bias + user_bias + item_bias + user_factors · item_factors
            global_bias = self.trainset.global_mean
            user_bias = self.model.bu[self.trainset.to_inner_uid(user_id)] if user_exists else 0
            item_bias = self.model.bi[self.trainset.to_inner_iid(item_id)] if item_exists else 0
            latent_score = np.dot(user_factors, item_factors)
            
            # Histórico do usuário para contexto
            user_profile = self.user_items.get(user_id, {})
            avg_user_rating = np.mean(list(user_profile.values())) if user_profile else global_bias
            
            return {
                'user_id': user_id,
                'item_id': item_id,
                'predicted_rating': float(prediction),
                'prediction_components': {
                    'global_bias': float(global_bias),
                    'user_bias': float(user_bias),
                    'item_bias': float(item_bias),
                    'latent_factors_score': float(latent_score),
                    'total': float(global_bias + user_bias + item_bias + latent_score)
                },
                'user_profile': {
                    'exists_in_training': user_exists,
                    'num_ratings': len(user_profile),
                    'avg_rating': float(avg_user_rating),
                    'rating_std': float(np.std(list(user_profile.values()))) if len(user_profile) > 1 else 0.0
                },
                'item_profile': {
                    'exists_in_training': item_exists,
                    'num_ratings': len(self.item_users.get(item_id, set())),
                    'popularity_rank': self._get_item_popularity_rank(item_id)
                },
                'top_contributing_factors': [
                    {
                        'factor_index': int(idx),
                        'user_value': float(user_factors[idx]),
                        'item_value': float(item_factors[idx]),
                        'contribution': float(factor_products[idx]),
                        'contribution_magnitude': float(factor_abs[idx])
                    }
                    for idx in top_factor_indices
                ],
                'model_info': {
                    'n_factors': self.n_factors,
                    'n_epochs': self.n_epochs,
                    'regularization': self.reg_all
                }
            }
            
        except Exception as e:
            return {
                'error': f'Erro ao explicar recomendação: {str(e)}',
                'user_id': user_id,
                'item_id': item_id
            }
    
    def _get_item_popularity_rank(self, item_id: int) -> int:
        """Retorna o ranking de popularidade do item (1 = mais popular)."""
        try:
            all_items = list(self.item_users.keys())
            item_popularity = [(iid, len(self.item_users.get(iid, set()))) 
                             for iid in all_items]
            item_popularity.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (iid, _) in enumerate(item_popularity, 1):
                if iid == item_id:
                    return rank
            return len(all_items)  # Item não encontrado = menos popular
        except:
            return -1
    
    def get_similar_users_by_factors(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        Encontra usuários similares baseado nos fatores latentes.
        
        Args:
            user_id: ID do usuário
            k: Número de usuários similares
            
        Returns:
            List[Tuple[int, float]]: Lista de (user_id, similarity)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            target_factors = self.get_user_factors(user_id)
            if np.allclose(target_factors, 0):
                return []
            
            similarities = []
            for uid in self.user_items.keys():
                if uid == user_id:
                    continue
                    
                other_factors = self.get_user_factors(uid)
                if not np.allclose(other_factors, 0):
                    # Similaridade do cosseno
                    similarity = np.dot(target_factors, other_factors) / (
                        np.linalg.norm(target_factors) * np.linalg.norm(other_factors)
                    )
                    similarities.append((uid, float(similarity)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            warnings.warn(f"Erro ao encontrar usuários similares para {user_id}: {e}")
            return []
    
    def get_similar_items_by_factors(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        Encontra itens similares baseado nos fatores latentes.
        
        Args:
            item_id: ID do item
            k: Número de itens similares
            
        Returns:
            List[Tuple[int, float]]: Lista de (item_id, similarity)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            target_factors = self.get_item_factors(item_id)
            if np.allclose(target_factors, 0):
                return []
            
            similarities = []
            for iid in self.item_users.keys():
                if iid == item_id:
                    continue
                    
                other_factors = self.get_item_factors(iid)
                if not np.allclose(other_factors, 0):
                    # Similaridade do cosseno
                    similarity = np.dot(target_factors, other_factors) / (
                        np.linalg.norm(target_factors) * np.linalg.norm(other_factors)
                    )
                    similarities.append((iid, float(similarity)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            warnings.warn(f"Erro ao encontrar itens similares para {item_id}: {e}")
            return []
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas detalhadas do modelo treinado.
        
        Returns:
            Dict com estatísticas do modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            # Análise dos fatores latentes
            user_factors_norm = np.linalg.norm(self.model.pu, axis=1)
            item_factors_norm = np.linalg.norm(self.model.qi, axis=1)
            
            # Análise dos bias
            user_bias_stats = {
                'mean': float(np.mean(self.model.bu)),
                'std': float(np.std(self.model.bu)),
                'min': float(np.min(self.model.bu)),
                'max': float(np.max(self.model.bu))
            }
            
            item_bias_stats = {
                'mean': float(np.mean(self.model.bi)),
                'std': float(np.std(self.model.bi)),
                'min': float(np.min(self.model.bi)),
                'max': float(np.max(self.model.bi))
            }
            
            return {
                'model_parameters': {
                    'n_factors': self.n_factors,
                    'n_epochs': self.n_epochs,
                    'learning_rate': self.lr_all,
                    'regularization': self.reg_all
                },
                'training_info': {
                    'n_users': self.trainset.n_users,
                    'n_items': self.trainset.n_items,
                    'n_ratings': self.trainset.n_ratings,
                    'global_mean': float(self.trainset.global_mean),
                    'rating_scale': self.trainset.rating_scale,
                    'training_time': self.training_time
                },
                'factor_analysis': {
                    'user_factors_norm': {
                        'mean': float(np.mean(user_factors_norm)),
                        'std': float(np.std(user_factors_norm)),
                        'min': float(np.min(user_factors_norm)),
                        'max': float(np.max(user_factors_norm))
                    },
                    'item_factors_norm': {
                        'mean': float(np.mean(item_factors_norm)),
                        'std': float(np.std(item_factors_norm)),
                        'min': float(np.min(item_factors_norm)),
                        'max': float(np.max(item_factors_norm))
                    }
                },
                'bias_analysis': {
                    'user_bias': user_bias_stats,
                    'item_bias': item_bias_stats
                }
            }
            
        except Exception as e:
            return {'error': f'Erro ao calcular estatísticas: {str(e)}'}
    
    def validate_parameters(self) -> Dict[str, Any]:
        """
        Valida os parâmetros do modelo e sugere melhorias.
        
        Returns:
            Dict com validação e sugestões
        """
        validation = {
            'valid': True,
            'warnings': [],
            'suggestions': []
        }
        
        # Validação do número de fatores
        if self.n_factors < 10:
            validation['warnings'].append("Número de fatores muito baixo (< 10)")
            validation['suggestions'].append("Considere usar pelo menos 50 fatores")
        elif self.n_factors > 200:
            validation['warnings'].append("Número de fatores muito alto (> 200)")
            validation['suggestions'].append("Considere reduzir para 50-150 fatores")
        
        # Validação das épocas
        if self.n_epochs < 10:
            validation['warnings'].append("Número de épocas muito baixo (< 10)")
            validation['suggestions'].append("Considere usar pelo menos 20 épocas")
        elif self.n_epochs > 50:
            validation['warnings'].append("Número de épocas muito alto (> 50)")
            validation['suggestions'].append("Pode estar causando overfitting")
        
        # Validação da taxa de aprendizado
        if self.lr_all > 0.01:
            validation['warnings'].append("Taxa de aprendizado alta (> 0.01)")
            validation['suggestions'].append("Considere valores entre 0.001-0.005")
        elif self.lr_all < 0.001:
            validation['warnings'].append("Taxa de aprendizado baixa (< 0.001)")
            validation['suggestions'].append("Treinamento pode ser muito lento")
        
        # Validação da regularização
        if self.reg_all > 0.1:
            validation['warnings'].append("Regularização alta (> 0.1)")
            validation['suggestions'].append("Pode estar causando underfitting")
        elif self.reg_all < 0.001:
            validation['warnings'].append("Regularização baixa (< 0.001)")
            validation['suggestions'].append("Pode estar causando overfitting")
        
        if validation['warnings']:
            validation['valid'] = False
        
        return validation
