# src/evaluation/evaluator.py - Versão Corrigida

import os
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
from ..models.base_model import BaseRecommender
from .metrics import Metrics


class Evaluator:
    """Avaliador corrigido para modelos de recomendação."""
    
    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o avaliador.
        
        Args:
            metrics_config: Configuração das métricas a calcular
        """
        self.metrics_config = metrics_config or {
            'rating_prediction': ['rmse', 'mae'],
            'ranking': ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k'],
            'beyond_accuracy': ['coverage', 'intra_list_diversity', 'novelty', 'personalization'],
            'k_values': [5, 10, 20],
            'relevance_threshold': 4.0  # NOVO: configurável
        }
        self.metrics = Metrics()
    
    def _validate_data(self, data: pd.DataFrame, required_cols: List[str]) -> None:
        """Valida se os dados têm as colunas necessárias."""
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
    
    def _get_relevance_threshold(self, dataset_scale: Optional[Tuple[int, int]] = None) -> float:
        """
        Determina threshold de relevância baseado na escala do dataset.
        
        Args:
            dataset_scale: Tupla (min_rating, max_rating)
            
        Returns:
            float: Threshold de relevância
        """
        if dataset_scale:
            min_rating, max_rating = dataset_scale
            # Usa 80% da escala como threshold
            return min_rating + 0.8 * (max_rating - min_rating)
        
        return self.metrics_config.get('relevance_threshold', 4.0)
    
    def evaluate_rating_prediction(self, 
                                 model: BaseRecommender, 
                                 test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Avalia métricas de predição de rating com validações.
        """
        self._validate_data(test_data, ['user_id', 'item_id', 'rating'])
        
        results = {}
        
        # Prepara dados para predição
        test_pairs = [(row['user_id'], row['item_id']) for _, row in test_data.iterrows()]
        true_ratings = test_data['rating'].values
        
        # Faz predições com tratamento de erro
        print("Fazendo predições...")
        start_time = time.time()
        try:
            predictions = model.predict_batch(test_pairs)
        except Exception as e:
            warnings.warn(f"Erro ao fazer predições: {e}")
            return {'error': str(e)}
        
        prediction_time = time.time() - start_time
        
        # Remove predições inválidas (NaN, inf)
        valid_mask = np.isfinite(predictions) & np.isfinite(true_ratings)
        if not valid_mask.all():
            warnings.warn(f"Removendo {(~valid_mask).sum()} predições inválidas")
            predictions = predictions[valid_mask]
            true_ratings = true_ratings[valid_mask]
        
        if len(predictions) == 0:
            return {'error': 'Nenhuma predição válida'}
        
        # Calcula métricas
        if 'rmse' in self.metrics_config['rating_prediction']:
            results['rmse'] = self.metrics.rmse(true_ratings, predictions)
        
        if 'mae' in self.metrics_config['rating_prediction']:
            results['mae'] = self.metrics.mae(true_ratings, predictions)
        
        results['prediction_time'] = prediction_time
        results['predictions_per_second'] = len(test_pairs) / prediction_time
        results['valid_predictions'] = len(predictions)
        results['total_predictions'] = len(test_pairs)
        
        return results
    
    def evaluate_ranking(self,
                        model: BaseRecommender,
                        test_data: pd.DataFrame,
                        train_data: pd.DataFrame,
                        n_recommendations: int = 10,
                        dataset_scale: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Avalia métricas de ranking com correções.
        """
        self._validate_data(test_data, ['user_id', 'item_id', 'rating'])
        self._validate_data(train_data, ['user_id', 'item_id', 'rating'])
        
        results = {}
        relevance_threshold = self._get_relevance_threshold(dataset_scale)
        
        # Agrupa dados de teste por usuário
        test_by_user = test_data.groupby('user_id').agg({
            'item_id': list,
            'rating': list
        })
        
        # Gera recomendações para cada usuário
        recommendations_dict = {}
        relevance_dict = {}
        relevance_scores_dict = {}  # Para NDCG
        
        print("Gerando recomendações...")
        failed_users = 0
        
        for user_id in tqdm(test_by_user.index):
            try:
                # Gera recomendações
                recs = model.recommend(user_id, n_recommendations, exclude_seen=True)
                if not recs:  # Lista vazia
                    failed_users += 1
                    continue
                    
                recommendations_dict[user_id] = [item_id for item_id, _ in recs]
                
                # Define itens relevantes
                test_items = test_by_user.loc[user_id, 'item_id']
                test_ratings = test_by_user.loc[user_id, 'rating']
                
                # Relevância binária
                relevant_items = [item for item, rating in zip(test_items, test_ratings) 
                                if rating >= relevance_threshold]
                relevance_dict[user_id] = relevant_items
                
                # Scores de relevância para NDCG (normalizado 0-1)
                if dataset_scale:
                    min_r, max_r = dataset_scale
                    normalized_scores = {item: (rating - min_r) / (max_r - min_r) 
                                       for item, rating in zip(test_items, test_ratings)}
                else:
                    normalized_scores = {item: rating for item, rating in zip(test_items, test_ratings)}
                
                relevance_scores_dict[user_id] = normalized_scores
                
            except Exception as e:
                failed_users += 1
                continue
        
        if failed_users > 0:
            warnings.warn(f"Falha ao gerar recomendações para {failed_users} usuários")
        
        if not recommendations_dict:
            return {'error': 'Nenhuma recomendação gerada com sucesso'}
        
        # Calcula métricas para diferentes valores de k
        for k in self.metrics_config['k_values']:
            k_results = {}
            
            # Precision@k e Recall@k
            precisions = []
            recalls = []
            ndcgs = []
            
            for user_id, recommended in recommendations_dict.items():
                if user_id in relevance_dict:
                    relevant = relevance_dict[user_id]
                    
                    precision = self.metrics.precision_at_k(recommended, relevant, k)
                    recall = self.metrics.recall_at_k(recommended, relevant, k)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    
                    # NDCG@k
                    if 'ndcg_at_k' in self.metrics_config['ranking']:
                        relevance_scores = relevance_scores_dict.get(user_id, {})
                        ndcg = self.metrics.ndcg_at_k(recommended, relevance_scores, k)
                        ndcgs.append(ndcg)
            
            k_results['precision'] = np.mean(precisions) if precisions else 0.0
            k_results['recall'] = np.mean(recalls) if recalls else 0.0
            
            if ndcgs:
                k_results['ndcg'] = np.mean(ndcgs)
            
            # MAP@k
            if 'map_at_k' in self.metrics_config['ranking']:
                k_results['map'] = self.metrics.map_at_k(recommendations_dict, relevance_dict, k)
            
            results[f'at_{k}'] = k_results
        
        results['evaluated_users'] = len(recommendations_dict)
        results['failed_users'] = failed_users
        results['relevance_threshold'] = relevance_threshold
        
        return results
    
    def evaluate_beyond_accuracy(self,
                               model: BaseRecommender,
                               train_data: pd.DataFrame,
                               n_users_sample: int = 100,
                               n_recommendations: int = 10) -> Dict[str, float]:
        """
        Avalia métricas além da acurácia com correções.
        """
        self._validate_data(train_data, ['user_id', 'item_id', 'rating'])
        
        results = {}
        
        # Amostra usuários
        all_users = train_data['user_id'].unique()
        sample_size = min(n_users_sample, len(all_users))
        sample_users = np.random.choice(all_users, sample_size, replace=False)
        
        # Todos os itens
        all_items = set(train_data['item_id'].unique())
        
        # Gera recomendações
        recommendations_dict = {}
        failed_users = 0
        
        for user_id in tqdm(sample_users, desc="Gerando recomendações para análise"):
            try:
                recs = model.recommend(user_id, n_recommendations, exclude_seen=True)
                if recs:
                    recommendations_dict[user_id] = [item_id for item_id, _ in recs]
            except Exception:
                failed_users += 1
                continue
        
        if not recommendations_dict:
            return {'error': 'Nenhuma recomendação gerada'}
        
        # Cobertura
        if 'coverage' in self.metrics_config['beyond_accuracy']:
            results['coverage'] = self.metrics.coverage(recommendations_dict, all_items)
        
        # Personalização
        if 'personalization' in self.metrics_config['beyond_accuracy']:
            results['personalization'] = self.metrics.personalization(recommendations_dict)
        
        # Popularidade dos itens (normalizada)
        item_counts = train_data['item_id'].value_counts()
        total_interactions = len(train_data)
        item_popularity = (item_counts / total_interactions).to_dict()
        
        # Novidade
        if 'novelty' in self.metrics_config['beyond_accuracy']:
            results['novelty'] = self.metrics.novelty(recommendations_dict, item_popularity)
        
        results['evaluated_users'] = len(recommendations_dict)
        results['failed_users'] = failed_users

        # Diversidade intra-lista melhorada
        if 'intra_list_diversity' in self.metrics_config['beyond_accuracy']:
            item_features = getattr(model, 'item_features', None)
            if item_features:
                diversity_scores = []
                for recommended in recommendations_dict.values():
                    diversity = self.metrics.intra_list_diversity_enhanced(
                        recommended, item_features
                    )
                    diversity_scores.append(diversity)
                results['intra_list_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
            else:
                # Fallback para método original
                diversity_scores = []
                for recommended in recommendations_dict.values():
                    diversity = len(set(recommended)) / len(recommended) if recommended else 0.0
                    diversity_scores.append(diversity)
                results['intra_list_diversity'] = np.mean(diversity_scores)

        # Serendipidade
        if 'serendipity' in self.metrics_config['beyond_accuracy']:
            # Precisa do histórico dos usuários (dados de treino por usuário)
            user_profiles = train_data.groupby('user_id')['item_id'].apply(list).to_dict()
            
            # Filtra apenas usuários da amostra
            sample_user_profiles = {uid: user_profiles.get(uid, []) 
                                  for uid in recommendations_dict.keys() 
                                  if uid in user_profiles}
            
            item_features = getattr(model, 'item_features', None)
            
            results['serendipity'] = self.metrics.serendipity(
                recommendations_dict,
                sample_user_profiles,
                item_features
            )

        # Cobertura de diversidade (exemplo com gêneros)
        if hasattr(model, 'item_features') and model.item_features:
            # Tenta encontrar features categóricas comuns
            sample_features = next(iter(model.item_features.values()))
            for feature_name in ['genre', 'category', 'type', 'class']:
                if feature_name in sample_features:
                    coverage = self.metrics.diversity_coverage(
                        recommendations_dict, model.item_features, feature_name
                    )
                    results[f'diversity_coverage_{feature_name}'] = coverage
                    break
        
        return results
    
    def evaluate_model(self,
                      model: BaseRecommender,
                      train_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      dataset_scale: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Avaliação completa de um modelo com tratamento de erros.
        """
        print(f"\nAvaliando modelo: {model.model_name}")
        
        all_results = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'model_params': model.params,
            'training_time': getattr(model, 'training_time', None),
            'evaluation_timestamp': time.time()
        }
        
        # Métricas de predição de rating
        if self.metrics_config.get('rating_prediction'):
            print("Calculando métricas de predição...")
            try:
                rating_metrics = self.evaluate_rating_prediction(model, test_data)
                all_results.update(rating_metrics)
            except Exception as e:
                all_results['rating_prediction_error'] = str(e)
                warnings.warn(f"Erro ao calcular métricas de predição: {e}")
        
        # Métricas de ranking
        if self.metrics_config.get('ranking'):
            print("Calculando métricas de ranking...")
            try:
                ranking_metrics = self.evaluate_ranking(
                    model, test_data, train_data, dataset_scale=dataset_scale
                )
                all_results['ranking'] = ranking_metrics
            except Exception as e:
                all_results['ranking_error'] = str(e)
                warnings.warn(f"Erro ao calcular métricas de ranking: {e}")
        
        # Métricas além da acurácia
        if self.metrics_config.get('beyond_accuracy'):
            print("Calculando métricas de diversidade...")
            try:
                beyond_metrics = self.evaluate_beyond_accuracy(model, train_data)
                all_results.update(beyond_metrics)
            except Exception as e:
                all_results['beyond_accuracy_error'] = str(e)
                warnings.warn(f"Erro ao calcular métricas além da acurácia: {e}")
        
        return all_results
    
    def load_item_features(self, 
                          features_file: Optional[str] = None,
                          dataset_name: str = 'movielens') -> Optional[Dict[int, Dict[str, any]]]:
        """
        Carrega features dos itens para cálculo de diversidade e serendipidade.
        
        Args:
            features_file: Caminho para arquivo de features
            dataset_name: Nome do dataset para features padrão
            
        Returns:
            Dict com features dos itens ou None
        """
        if features_file and os.path.exists(features_file):
            # Carrega de arquivo personalizado
            return self._load_features_from_file(features_file)
        
        # Features padrão por dataset
        if dataset_name.startswith('movielens'):
            return self._load_movielens_features()
        elif dataset_name == 'lastfm':
            return self._load_lastfm_features()
        
        return None
    
    def _load_movielens_features(self) -> Dict[int, Dict[str, any]]:
        """Carrega features do MovieLens (gêneros dos filmes)."""
        try:
            import os
            movies_file = None
            
            # Procura arquivo de filmes
            for path in ['data/raw/movies.csv', 'data/raw/movies.dat', 'data/processed/movies.csv']:
                if os.path.exists(path):
                    movies_file = path
                    break
            
            if not movies_file:
                return {}
            
            if movies_file.endswith('.csv'):
                movies_df = pd.read_csv(movies_file)
            else:  # .dat
                movies_df = pd.read_csv(movies_file, sep='::', engine='python', 
                                     names=['movieId', 'title', 'genres'])
            
            item_features = {}
            for _, row in movies_df.iterrows():
                item_id = row['movieId'] if 'movieId' in row else row['item_id']
                genres = row['genres'].split('|') if pd.notna(row['genres']) else []
                
                item_features[item_id] = {
                    'genre': genres,
                    'title': row.get('title', ''),
                    'year': self._extract_year(row.get('title', ''))
                }
            
            return item_features
            
        except Exception as e:
            warnings.warn(f"Erro ao carregar features do MovieLens: {e}")
            return {}
    
    def _extract_year(self, title: str) -> Optional[int]:
        """Extrai ano do título do filme."""
        import re
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    
    def _load_features_from_file(self, filepath: str) -> Dict[int, Dict[str, any]]:
        """Carrega features de arquivo CSV personalizado."""
        try:
            df = pd.read_csv(filepath)
            features = {}
            
            for _, row in df.iterrows():
                item_id = row['item_id']
                features[item_id] = row.drop('item_id').to_dict()
            
            return features
            
        except Exception as e:
            warnings.warn(f"Erro ao carregar features de {filepath}: {e}")
            return {}