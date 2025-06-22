# src/evaluation/evaluator.py

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from ..models.base_model import BaseRecommender
from .metrics import Metrics


class Evaluator:
    """Avaliador principal para modelos de recomendação."""
    
    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o avaliador.
        
        Args:
            metrics_config: Configuração das métricas a calcular
        """
        self.metrics_config = metrics_config or {
            'rating_prediction': ['rmse', 'mae'],
            'ranking': ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k'],
            'beyond_accuracy': ['coverage', 'diversity', 'novelty'],
            'k_values': [5, 10, 20]
        }
        self.metrics = Metrics()
    
    def evaluate_rating_prediction(self, 
                                 model: BaseRecommender, 
                                 test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Avalia métricas de predição de rating.
        
        Args:
            model: Modelo treinado
            test_data: Dados de teste
            
        Returns:
            Dict com métricas calculadas
        """
        results = {}
        
        # Prepara dados para predição
        test_pairs = [(row['user_id'], row['item_id']) for _, row in test_data.iterrows()]
        true_ratings = test_data['rating'].values
        
        # Faz predições
        print("Fazendo predições...")
        start_time = time.time()
        predictions = model.predict_batch(test_pairs)
        prediction_time = time.time() - start_time
        
        # Calcula métricas
        if 'rmse' in self.metrics_config['rating_prediction']:
            results['rmse'] = self.metrics.rmse(true_ratings, predictions)
        
        if 'mae' in self.metrics_config['rating_prediction']:
            results['mae'] = self.metrics.mae(true_ratings, predictions)
        
        results['prediction_time'] = prediction_time
        results['predictions_per_second'] = len(test_pairs) / prediction_time
        
        return results
    
    def evaluate_ranking(self,
                        model: BaseRecommender,
                        test_data: pd.DataFrame,
                        train_data: pd.DataFrame,
                        n_recommendations: int = 10) -> Dict[str, Any]:
        """
        Avalia métricas de ranking.
        
        Args:
            model: Modelo treinado
            test_data: Dados de teste
            train_data: Dados de treino
            n_recommendations: Número de recomendações por usuário
            
        Returns:
            Dict com métricas calculadas
        """
        results = {}
        
        # Agrupa dados de teste por usuário
        test_by_user = test_data.groupby('user_id').agg({
            'item_id': list,
            'rating': list
        })
        
        # Itens vistos no treino por usuário
        train_by_user = train_data.groupby('user_id')['item_id'].apply(set).to_dict()
        
        # Gera recomendações para cada usuário
        recommendations_dict = {}
        relevance_dict = {}
        
        print("Gerando recomendações...")
        for user_id in tqdm(test_by_user.index):
            # Gera recomendações
            recs = model.recommend(user_id, n_recommendations, exclude_seen=True)
            recommendations_dict[user_id] = [item_id for item_id, _ in recs]
            
            # Define itens relevantes (rating >= 4)
            test_items = test_by_user.loc[user_id, 'item_id']
            test_ratings = test_by_user.loc[user_id, 'rating']
            
            relevant_items = [item for item, rating in zip(test_items, test_ratings) if rating >= 4]
            relevance_dict[user_id] = relevant_items
            
            # Scores de relevância para NDCG
            relevance_scores = {item: rating for item, rating in zip(test_items, test_ratings)}
        
        # Calcula métricas para diferentes valores de k
        for k in self.metrics_config['k_values']:
            k_results = {}
            
            # Precision@k e Recall@k
            precisions = []
            recalls = []
            
            for user_id, recommended in recommendations_dict.items():
                if user_id in relevance_dict:
                    relevant = relevance_dict[user_id]
                    
                    precision = self.metrics.precision_at_k(recommended, relevant, k)
                    recall = self.metrics.recall_at_k(recommended, relevant, k)
                    
                    precisions.append(precision)
                    recalls.append(recall)
            
            k_results['precision'] = np.mean(precisions) if precisions else 0.0
            k_results['recall'] = np.mean(recalls) if recalls else 0.0
            
            # MAP@k
            if 'map_at_k' in self.metrics_config['ranking']:
                k_results['map'] = self.metrics.map_at_k(recommendations_dict, relevance_dict, k)
            
            results[f'at_{k}'] = k_results
        
        return results
    
    def evaluate_beyond_accuracy(self,
                               model: BaseRecommender,
                               train_data: pd.DataFrame,
                               n_users_sample: int = 100) -> Dict[str, float]:
        """
        Avalia métricas além da acurácia (diversidade, cobertura, novidade).
        
        Args:
            model: Modelo treinado
            train_data: Dados de treino
            n_users_sample: Número de usuários para amostrar
            
        Returns:
            Dict com métricas calculadas
        """
        results = {}
        
        # Amostra usuários
        all_users = train_data['user_id'].unique()
        sample_users = np.random.choice(all_users, 
                                      min(n_users_sample, len(all_users)), 
                                      replace=False)
        
        # Todos os itens
        all_items = set(train_data['item_id'].unique())
        
        # Gera recomendações
        recommendations_dict = {}
        for user_id in sample_users:
            recs = model.recommend(user_id, 10, exclude_seen=True)
            recommendations_dict[user_id] = [item_id for item_id, _ in recs]
        
        # Cobertura
        if 'coverage' in self.metrics_config['beyond_accuracy']:
            results['coverage'] = self.metrics.coverage(recommendations_dict, all_items)
        
        # Popularidade dos itens
        item_counts = train_data['item_id'].value_counts()
        item_popularity = (item_counts / len(train_data)).to_dict()
        
        # Novidade
        if 'novelty' in self.metrics_config['beyond_accuracy']:
            results['novelty'] = self.metrics.novelty(recommendations_dict, item_popularity)
        
        return results
    
    def evaluate_model(self,
                      model: BaseRecommender,
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Avaliação completa de um modelo.
        
        Args:
            model: Modelo a avaliar
            train_data: Dados de treino
            test_data: Dados de teste
            
        Returns:
            Dict com todas as métricas calculadas
        """
        print(f"\nAvaliando modelo: {model.model_name}")
        
        all_results = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'model_params': model.params,
            'training_time': model.training_time
        }
        
        # Métricas de predição de rating
        if self.metrics_config.get('rating_prediction'):
            print("Calculando métricas de predição...")
            rating_metrics = self.evaluate_rating_prediction(model, test_data)
            all_results.update(rating_metrics)
        
        # Métricas de ranking
        if self.metrics_config.get('ranking'):
            print("Calculando métricas de ranking...")
            ranking_metrics = self.evaluate_ranking(model, test_data, train_data)
            all_results['ranking'] = ranking_metrics
        
        # Métricas além da acurácia
        if self.metrics_config.get('beyond_accuracy'):
            print("Calculando métricas de diversidade...")
            beyond_metrics = self.evaluate_beyond_accuracy(model, train_data)
            all_results.update(beyond_metrics)
        
        return all_results
    
    def compare_models(self, 
                      models: List[BaseRecommender],
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compara múltiplos modelos.
        
        Args:
            models: Lista de modelos para comparar
            train_data: Dados de treino
            test_data: Dados de teste
            
        Returns:
            pd.DataFrame: Comparação dos modelos
        """
        results_list = []
        
        for model in models:
            results = self.evaluate_model(model, train_data, test_data)
            
            # Flatten dos resultados para o DataFrame
            flat_results = {
                'model_name': results['model_name'],
                'model_type': results['model_type'],
                'training_time': results.get('training_time', None),
                'rmse': results.get('rmse', None),
                'mae': results.get('mae', None),
                'coverage': results.get('coverage', None),
                'novelty': results.get('novelty', None)
            }
            
            # Adiciona métricas de ranking
            if 'ranking' in results:
                for k_value, k_metrics in results['ranking'].items():
                    for metric_name, value in k_metrics.items():
                        flat_results[f'{metric_name}_{k_value}'] = value
            
            results_list.append(flat_results)
        
        return pd.DataFrame(results_list)
