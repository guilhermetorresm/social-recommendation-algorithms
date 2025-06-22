# src/utils/result_manager.py

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import pickle


class ResultManager:
    """Gerencia o salvamento e carregamento de resultados de experimentos."""
    
    def __init__(self, results_dir: str = 'data/results'):
        """
        Inicializa o gerenciador de resultados.
        
        Args:
            results_dir: Diretório para salvar os resultados
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_experiment_results(self, 
                              experiment_id: str,
                              model_name: str,
                              dataset_name: str,
                              metrics: Dict[str, Any],
                              predictions: Optional[pd.DataFrame] = None,
                              recommendations: Optional[Dict[int, List]] = None,
                              model_params: Optional[Dict[str, Any]] = None,
                              additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Salva os resultados completos de um experimento.
        
        Args:
            experiment_id: ID único do experimento
            model_name: Nome do modelo
            dataset_name: Nome do dataset
            metrics: Dicionário com as métricas calculadas
            predictions: DataFrame com predições (opcional)
            recommendations: Dicionário com recomendações por usuário (opcional)
            model_params: Parâmetros do modelo (opcional)
            additional_info: Informações adicionais (opcional)
            
        Returns:
            str: Caminho do diretório onde os resultados foram salvos
        """
        # Cria diretório para o experimento
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.join(
            self.results_dir, 
            f"{experiment_id}_{model_name}_{dataset_name}_{timestamp}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Metadados do experimento
        metadata = {
            'experiment_id': experiment_id,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'model_params': model_params or {},
            'additional_info': additional_info or {}
        }
        
        # Salva metadados
        with open(os.path.join(experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Salva métricas
        with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Salva métricas em CSV para fácil análise
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(
            os.path.join(experiment_dir, 'metrics.csv'), 
            index=False
        )
        
        # Salva predições se fornecidas
        if predictions is not None:
            predictions.to_csv(
                os.path.join(experiment_dir, 'predictions.csv'), 
                index=False
            )
        
        # Salva recomendações se fornecidas
        if recommendations is not None:
            with open(os.path.join(experiment_dir, 'recommendations.pkl'), 'wb') as f:
                pickle.dump(recommendations, f)
        
        # Cria um resumo consolidado
        summary = {
            **metadata,
            'metrics': metrics,
            'results_path': experiment_dir
        }
        
        # Adiciona ao log geral de experimentos
        self._update_experiments_log(summary)
        
        return experiment_dir
    
    def _update_experiments_log(self, summary: Dict[str, Any]) -> None:
        """Atualiza o log geral de experimentos."""
        log_path = os.path.join(self.results_dir, 'experiments_log.json')
        
        # Carrega log existente ou cria novo
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                experiments_log = json.load(f)
        else:
            experiments_log = []
        
        # Adiciona novo experimento
        experiments_log.append(summary)
        
        # Salva log atualizado
        with open(log_path, 'w') as f:
            json.dump(experiments_log, f, indent=2)
    
    def load_experiment_results(self, experiment_path: str) -> Dict[str, Any]:
        """
        Carrega os resultados de um experimento.
        
        Args:
            experiment_path: Caminho do diretório do experimento
            
        Returns:
            Dict contendo todos os dados do experimento
        """
        results = {}
        
        # Carrega metadados
        with open(os.path.join(experiment_path, 'metadata.json'), 'r') as f:
            results['metadata'] = json.load(f)
        
        # Carrega métricas
        with open(os.path.join(experiment_path, 'metrics.json'), 'r') as f:
            results['metrics'] = json.load(f)
        
        # Carrega predições se existirem
        predictions_path = os.path.join(experiment_path, 'predictions.csv')
        if os.path.exists(predictions_path):
            results['predictions'] = pd.read_csv(predictions_path)
        
        # Carrega recomendações se existirem
        recommendations_path = os.path.join(experiment_path, 'recommendations.pkl')
        if os.path.exists(recommendations_path):
            with open(recommendations_path, 'rb') as f:
                results['recommendations'] = pickle.load(f)
        
        return results
    
    def get_all_experiments(self) -> pd.DataFrame:
        """
        Retorna um DataFrame com todos os experimentos realizados.
        
        Returns:
            pd.DataFrame: DataFrame com informações de todos os experimentos
        """
        log_path = os.path.join(self.results_dir, 'experiments_log.json')
        
        if not os.path.exists(log_path):
            return pd.DataFrame()
        
        with open(log_path, 'r') as f:
            experiments = json.load(f)
        
        # Converte para DataFrame e expande métricas
        df_list = []
        for exp in experiments:
            flat_exp = {
                'experiment_id': exp['experiment_id'],
                'model_name': exp['model_name'],
                'dataset_name': exp['dataset_name'],
                'timestamp': exp['timestamp'],
                'datetime': exp['datetime'],
                'results_path': exp['results_path']
            }
            
            # Adiciona métricas ao registro flat
            for metric_name, metric_value in exp['metrics'].items():
                if isinstance(metric_value, dict):
                    for sub_metric, value in metric_value.items():
                        flat_exp[f"{metric_name}_{sub_metric}"] = value
                else:
                    flat_exp[metric_name] = metric_value
            
            df_list.append(flat_exp)
        
        return pd.DataFrame(df_list)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compara métricas de múltiplos experimentos.
        
        Args:
            experiment_ids: Lista de IDs de experimentos para comparar
            
        Returns:
            pd.DataFrame: DataFrame comparativo
        """
        all_experiments = self.get_all_experiments()
        
        # Filtra experimentos selecionados
        comparison_df = all_experiments[
            all_experiments['experiment_id'].isin(experiment_ids)
        ]
        
        return comparison_df.sort_values('datetime')
    
    def get_best_model(self, 
                      metric: str, 
                      dataset: Optional[str] = None,
                      minimize: bool = True) -> Dict[str, Any]:
        """
        Encontra o melhor modelo baseado em uma métrica.
        
        Args:
            metric: Nome da métrica
            dataset: Nome do dataset (opcional)
            minimize: Se True, menor é melhor; se False, maior é melhor
            
        Returns:
            Dict com informações do melhor modelo
        """
        all_experiments = self.get_all_experiments()
        
        if dataset:
            all_experiments = all_experiments[
                all_experiments['dataset_name'] == dataset
            ]
        
        if metric not in all_experiments.columns:
            raise ValueError(f"Métrica '{metric}' não encontrada")
        
        # Remove valores NaN
        valid_experiments = all_experiments.dropna(subset=[metric])
        
        if valid_experiments.empty:
            return {}
        
        # Encontra o melhor
        if minimize:
            best_idx = valid_experiments[metric].idxmin()
        else:
            best_idx = valid_experiments[metric].idxmax()
        
        return valid_experiments.loc[best_idx].to_dict()
