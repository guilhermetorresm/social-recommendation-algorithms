# experiments/run_experiments.py

import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import yaml
from typing import Dict, Any, List

from src.data.data_loader import MovieLens100KLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.baseline.global_mean import GlobalMeanModel
from src.models.baseline.popularity import PopularityModel
from src.models.collaborative.knn_user import KNNUserModel
from src.models.collaborative.knn_item import KNNItemModel
from src.models.collaborative.svd import SVDModel
from src.models.content.content_based import ContentBasedModel

from src.evaluation.evaluator import Evaluator
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.result_manager import ResultManager


class ExperimentRunner:
    """Classe principal para executar experimentos de recomendação."""
    
    def __init__(self, config_path: str = None):
        """
        Inicializa o executor de experimentos.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = Config(config_path)
        print("Configuração de métricas:", self.config.get('evaluation.metrics'))
        self.logger = Logger('experiment_runner')
        self.result_manager = ResultManager(self.config.get('data.results_path'))
        self.evaluator = Evaluator(self.config.get('evaluation.metrics'))
        
    def load_and_prepare_data(self, dataset_name: str = 'movielens-100k') -> tuple:
        """
        Carrega e prepara os dados para experimento.
        
        Args:
            dataset_name: Nome do dataset
            
        Returns:
            tuple: (train_data, test_data, dataset_info)
        """
        self.logger.info(f"Carregando dataset: {dataset_name}")
        
        # Carrega dados
        if dataset_name == 'movielens-100k':
            loader = MovieLens100KLoader(self.config.get('data.raw_path'))
            data = loader.load()
            items_data = loader.load_items()
            dataset_info = loader.get_metadata()
        else:
            raise ValueError(f"Dataset não suportado: {dataset_name}")
        
        # Pré-processa dados
        preprocessor = DataPreprocessor(
            min_user_ratings=5,
            min_item_ratings=5
        )
        data_filtered = preprocessor.filter_data(data)
        
        # Divide dados
        splitter = DataSplitter(
            test_size=1 - self.config.get('evaluation.train_test_split'),
            random_state=self.config.get('evaluation.random_seed')
        )
        train_data, test_data = splitter.random_split(data_filtered)
        
        # Log informações
        split_info = splitter.get_split_info(train_data, test_data)
        self.logger.info(f"Divisão dos dados: {split_info}")
        
        return train_data, test_data, dataset_info, items_data
    
    def get_available_models(self) -> Dict[str, Any]:
        """Retorna modelos disponíveis para experimento."""
        models = {
            'global_mean': {
                'class': GlobalMeanModel,
                'params': {}
            },
            'popularity_count': {
                'class': PopularityModel,
                'params': {'popularity_metric': 'rating_count'}
            },
            'popularity_rating': {
                'class': PopularityModel,
                'params': {'popularity_metric': 'mean_rating'}
            },
            'knn_user': {
                'class': KNNUserModel,
                'params': self.config.get('models.default_params.knn', {})
            },
            'knn_item': {
                'class': KNNItemModel,
                'params': self.config.get('models.default_params.knn', {})
            },
            'svd': {
                'class': SVDModel,
                'params': self.config.get('models.default_params.svd', {})
            },
            'content_based': {
                'class': ContentBasedModel,
                # 'params': self.config.get('models.default_params.content_based', {})
                'params': {}
            }
        }
        
        # Adicione aqui outros modelos conforme implementados
        # Por exemplo:
        # 'knn_user': {
        #     'class': KNNUserModel,
        #     'params': self.config.get('models.default_params.knn', {})
        # }
        
        return models
    
    def run_single_experiment(self, 
                            model_name: str,
                            train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            items_data: pd.DataFrame,
                            dataset_name: str,
                            experiment_id: str) -> Dict[str, Any]:
        """
        Executa um único experimento.
        
        Args:
            model_name: Nome do modelo
            train_data: Dados de treino
            test_data: Dados de teste
            dataset_name: Nome do dataset
            experiment_id: ID do experimento
            
        Returns:
            Dict com resultados do experimento
        """
        self.logger.experiment_start(f"{experiment_id}_{model_name}", {
            'model': model_name,
            'dataset': dataset_name,
            'train_size': len(train_data),
            'test_size': len(test_data)
        })
        
        # Obtém configuração do modelo
        available_models = self.get_available_models()
        if model_name not in available_models:
            raise ValueError(f"Modelo não encontrado: {model_name}")
        
        model_config = available_models[model_name]
        
        # Cria e treina modelo
        model = model_config['class'](**model_config['params'])
        
        fit_kwargs = {}
        if model_name == 'content_based':
            fit_kwargs['items_data'] = items_data  # Passa os dados dos itens para o modelo Content-Based
        model.fit(train_data, **fit_kwargs)
        
        # Avalia modelo
        results = self.evaluator.evaluate_model(model, train_data, test_data)
        
        # Salva resultados
        result_path = self.result_manager.save_experiment_results(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=results,
            model_params=model_config['params']
        )
        
        # Salva modelo
        model_path = os.path.join(result_path, 'model.pkl')
        model.save_model(model_path)
        
        self.logger.experiment_end(f"{experiment_id}_{model_name}", results)
        
        return results
    
    def run_all_baselines(self, dataset_name: str = 'movielens-100k'):
        """Executa todos os modelos baseline."""
        # Gera ID único para o conjunto de experimentos
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Carrega dados
        train_data, test_data, dataset_info, items_data = self.load_and_prepare_data(dataset_name)
        
        # Modelos baseline para executar
        baseline_models = ['global_mean', 'popularity_count', 'popularity_rating']
        
        results = {}
        for model_name in baseline_models:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Executando modelo: {model_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                result = self.run_single_experiment(
                    model_name=model_name,
                    train_data=train_data,
                    test_data=test_data,
                    items_data=items_data,  # Passa os dados dos itens para o modelo Content-Based
                    dataset_name=dataset_name,
                    experiment_id=experiment_id
                )
                results[model_name] = result
                
            except Exception as e:
                self.logger.error(f"Erro ao executar {model_name}: {str(e)}")
                continue
        
        # Gera relatório comparativo
        self._generate_comparison_report(results, experiment_id)
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, Any], experiment_id: str):
        """Gera relatório comparativo dos modelos."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RELATÓRIO COMPARATIVO DOS MODELOS")
        self.logger.info("="*70)
        
        # Cria DataFrame para comparação
        comparison_data = []
        for model_name, model_results in results.items():
            row = {
                'Modelo': model_name,
                'RMSE': model_results.get('rmse', '-'),
                'MAE': model_results.get('mae', '-'),
                'Tempo Treino (s)': f"{model_results.get('training_time', 0):.3f}",
                'Cobertura': f"{model_results.get('coverage', 0)*100:.1f}%",
                'Novidade': f"{model_results.get('novelty', 0):.3f}"
            }
            
            # Adiciona métricas de ranking @10
            if 'ranking' in model_results and 'at_10' in model_results['ranking']:
                at_10 = model_results['ranking']['at_10']
                row['Precision@10'] = f"{at_10.get('precision', 0):.3f}"
                row['Recall@10'] = f"{at_10.get('recall', 0):.3f}"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Salva relatório
        report_path = os.path.join(
            self.config.get('data.results_path'),
            f'comparison_report_{experiment_id}.csv'
        )
        comparison_df.to_csv(report_path, index=False)
        self.logger.info(f"\nRelatório salvo em: {report_path}")
    
    def run_custom_experiment(self, config_file: str):
        """
        Executa experimento customizado baseado em arquivo de configuração.
        
        Args:
            config_file: Caminho para arquivo YAML com configuração do experimento
        """
        with open(config_file, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        # Implementar lógica para experimentos customizados
        pass

    def run_all_experiments(self, dataset_name: str = 'movielens-100k'):
        """Executa todos os modelos"""
        # Gera ID único para o conjunto de experimentos
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Carrega dados
        train_data, test_data, dataset_info, items_data = self.load_and_prepare_data(dataset_name)
        
        # Modelos baseline para executar
        baseline_models = ['global_mean', 'popularity_count', 'popularity_rating', 'knn_user', 'knn_item', 'svd', 'content_based']
        
        results = {}
        for model_name in baseline_models:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Executando modelo: {model_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                result = self.run_single_experiment(
                    model_name=model_name,
                    train_data=train_data,
                    test_data=test_data,
                    items_data=items_data,  # Passa os dados dos itens para o modelo Content-Based
                    dataset_name=dataset_name,
                    experiment_id=experiment_id
                )
                results[model_name] = result
                
            except Exception as e:
                self.logger.error(f"Erro ao executar {model_name}: {str(e)}")
                continue
        
        # Gera relatório comparativo
        self._generate_comparison_report(results, experiment_id)
        
        return results

    def run_knn_explorer(self, dataset_name: str = 'movielens-100k'):
        """
        Executa análise exploratória dos modelos KNN com diferentes configurações.
        
        Args:
            dataset_name: Nome do dataset
        """
        # Gera ID único para o conjunto de experimentos
        experiment_id = f"knn_explorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Carrega dados
        train_data, test_data, dataset_info, items_data = self.load_and_prepare_data(dataset_name)
        
        # Parâmetros para exploração
        k_values = [10, 20, 30, 40, 50]
        sim_metrics = ['cosine', 'msd', 'pearson']
        min_supports = [1, 3, 5]
        
        # Modelos KNN para testar
        knn_models = ['knn_user', 'knn_item']
        
        results = {}
        total_experiments = len(knn_models) * len(k_values) * len(sim_metrics) * len(min_supports)
        current_exp = 0
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"INICIANDO EXPLORAÇÃO KNN - {total_experiments} experimentos")
        self.logger.info(f"{'='*70}")
        
        for model_type in knn_models:
            results[model_type] = {}
            
            for k in k_values:
                for sim_metric in sim_metrics:
                    for min_support in min_supports:
                        current_exp += 1
                        config_key = f"k{k}_{sim_metric}_ms{min_support}"
                        
                        self.logger.info(f"\n[{current_exp}/{total_experiments}] {model_type}: k={k}, sim={sim_metric}, min_support={min_support}")
                        
                        try:
                            # Configura parâmetros do modelo
                            model_params = {
                                'k': k,
                                'sim_metric': sim_metric,
                                'min_support': min_support
                            }
                            
                            # Executa experimento
                            result = self.run_single_knn_experiment(
                                model_type=model_type,
                                model_params=model_params,
                                train_data=train_data,
                                test_data=test_data,
                                dataset_name=dataset_name,
                                experiment_id=experiment_id,
                                config_key=config_key
                            )
                            
                            results[model_type][config_key] = {
                                'params': model_params,
                                'metrics': result
                            }
                            
                        except Exception as e:
                            self.logger.error(f"Erro em {model_type} {config_key}: {str(e)}")
                            results[model_type][config_key] = {
                                'params': model_params,
                                'error': str(e)
                            }
                            continue
        
        # Gera relatórios de análise
        self._generate_knn_analysis_report(results, experiment_id)
        
        return results

    def run_single_knn_experiment(self, 
                                model_type: str,
                                model_params: Dict[str, Any],
                                train_data: pd.DataFrame,
                                test_data: pd.DataFrame,
                                dataset_name: str,
                                experiment_id: str,
                                config_key: str) -> Dict[str, Any]:
        """
        Executa um único experimento KNN.
        
        Args:
            model_type: Tipo do modelo KNN ('knn_user' ou 'knn_item')
            model_params: Parâmetros do modelo
            train_data: Dados de treino
            test_data: Dados de teste
            dataset_name: Nome do dataset
            experiment_id: ID do experimento
            config_key: Chave da configuração
            
        Returns:
            Dict com resultados do experimento
        """
        model_name = f"{model_type}_{config_key}"
        
        self.logger.experiment_start(f"{experiment_id}_{model_name}", {
            'model': model_type,
            'params': model_params,
            'dataset': dataset_name,
            'train_size': len(train_data),
            'test_size': len(test_data)
        })
        
        # Obtém classe do modelo
        available_models = self.get_available_models()
        if model_type not in available_models:
            raise ValueError(f"Modelo não encontrado: {model_type}")
        
        model_class = available_models[model_type]['class']
        
        # Cria e treina modelo com parâmetros específicos
        model = model_class(**model_params)
        model.fit(train_data)
        
        # Avalia modelo
        results = self.evaluator.evaluate_model(model, train_data, test_data)
        
        # Salva resultados
        result_path = self.result_manager.save_experiment_results(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=results,
            model_params=model_params
        )
        
        # Salva modelo
        model_path = os.path.join(result_path, 'model.pkl')
        model.save_model(model_path)
        
        self.logger.experiment_end(f"{experiment_id}_{model_name}", results)
        
        return results

    def _generate_knn_analysis_report(self, results: Dict[str, Any], experiment_id: str):
        """
        Gera relatório de análise dos experimentos KNN.
        
        Args:
            results: Resultados dos experimentos
            experiment_id: ID do experimento
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("RELATÓRIO DE ANÁLISE KNN")
        self.logger.info("="*70)
        
        for model_type, model_results in results.items():
            self.logger.info(f"\n{'-'*50}")
            self.logger.info(f"ANÁLISE: {model_type.upper()}")
            self.logger.info(f"{'-'*50}")
            
            # Prepara dados para análise
            analysis_data = []
            for config_key, config_result in model_results.items():
                if 'error' in config_result:
                    continue
                    
                params = config_result['params']
                metrics = config_result['metrics']
                
                row = {
                    'K': params['k'],
                    'Similaridade': params['sim_metric'],
                    'Min_Support': params['min_support'],
                    'RMSE': metrics.get('rmse', '-'),
                    'MAE': metrics.get('mae', '-'),
                    'Tempo_Treino': f"{metrics.get('training_time', 0):.3f}",
                    'Cobertura': f"{metrics.get('coverage', 0)*100:.1f}%"
                }
                
                # Adiciona métricas de ranking se disponíveis
                if 'ranking' in metrics and 'at_10' in metrics['ranking']:
                    at_10 = metrics['ranking']['at_10']
                    row['Precision@10'] = f"{at_10.get('precision', 0):.3f}"
                    row['Recall@10'] = f"{at_10.get('recall', 0):.3f}"
                
                analysis_data.append(row)
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                
                # Ordena por RMSE
                if 'RMSE' in analysis_df.columns:
                    analysis_df = analysis_df.sort_values('RMSE')
                
                print(f"\nMelhores configurações para {model_type}:")
                print(analysis_df.head(10).to_string(index=False))
                
                # Salva relatório detalhado
                report_path = os.path.join(
                    self.config.get('data.results_path'),
                    f'knn_analysis_{model_type}_{experiment_id}.csv'
                )
                analysis_df.to_csv(report_path, index=False)
                self.logger.info(f"Relatório {model_type} salvo em: {report_path}")
            else:
                self.logger.warning(f"Nenhum resultado válido para {model_type}")
        
        # Análise comparativa entre user-based e item-based
        self._generate_knn_comparison_analysis(results, experiment_id)
    
    def run_content_based_only(self, dataset_name: str = 'movielens-100k'):
        """
        Executa o experimento apenas para o modelo Content-Based.
        """
        self.logger.info(f"--- INICIANDO EXECUÇÃO DEDICADA: CONTENT-BASED (dataset: {dataset_name}) ---")
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S') + "_content_based"
        
        # Carrega os dados, incluindo items_data, que é essencial para este modelo
        train_data, test_data, dataset_info, items_data = self.load_and_prepare_data(dataset_name)
        
        model_name = 'content_based'
        
        self.logger.info(f"Executando modelo: {model_name}")
        try:
            # Chama o executor de experimento único, passando todos os dados necessários
            result = self.run_single_experiment(
                model_name=model_name,
                train_data=train_data,
                test_data=test_data,
                items_data=items_data, # Passa os dados dos itens para o método
                dataset_name=dataset_name,
                experiment_id=experiment_id
            )
            # Imprime os resultados no log para visualização imediata
            # self.logger.info(f"Resultados para {model_name}:")
            # for metric, value in result['metrics'].items():
            #     self.logger.info(f"  - {metric}: {value:.4f}")

        except Exception as e:
            self.logger.error(f"Erro ao executar o modelo {model_name}: {str(e)}")
        
        self.logger.info("--- EXECUÇÃO DEDICADA CONTENT-BASED CONCLUÍDA ---")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Executar experimentos de sistemas de recomendação')
    parser.add_argument('--mode', type=str, default='baselines',
                       choices=['baselines', 'all', 'custom', 'knn_explorer', 'content_based'],
                       help='Modo de execução')
    parser.add_argument('--dataset', type=str, default='movielens-100k',
                       help='Dataset a utilizar')
    parser.add_argument('--config', type=str, default=None,
                       help='Arquivo de configuração')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Lista de modelos específicos para executar')
    
    args = parser.parse_args()
    
    # Cria executor
    runner = ExperimentRunner(args.config)
    
    # Executa experimentos
    if args.mode == 'baselines':
        runner.run_all_baselines(args.dataset)
    elif args.mode == 'knn_explorer':
        runner.run_knn_explorer(args.dataset)
    elif args.mode == 'all':
        runner.run_all_experiments(args.dataset)
    elif args.mode == 'content_based': # <-- ADICIONADO ESTE BLOCO
        runner.run_content_based_only(args.dataset)
    elif args.mode == 'custom' and args.config:
        runner.run_custom_experiment(args.config)
    else:
        print("Modo de execução inválido ou configuração faltando")


if __name__ == "__main__":
    main()
