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
        model.fit(train_data)
        
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


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Executar experimentos de sistemas de recomendação')
    parser.add_argument('--mode', type=str, default='baselines',
                       choices=['baselines', 'all', 'custom'],
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
    elif args.mode == 'all':
        # Implementar execução de todos os modelos
        pass
    elif args.mode == 'custom' and args.config:
        runner.run_custom_experiment(args.config)
    else:
        print("Modo de execução inválido ou configuração faltando")


if __name__ == "__main__":
    main()
