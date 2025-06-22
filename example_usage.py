# example_usage.py
"""
Script de exemplo demonstrando como usar o sistema de recomendação.
Execute este script para testar rapidamente o sistema.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Importações do projeto
from src.data.data_loader import MovieLens100KLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.baseline.global_mean import GlobalMeanModel
from src.models.baseline.popularity import PopularityModel
from src.models.collaborative.knn_user import KNNUserModel
from src.models.collaborative.knn_item import KNNItemModel
from src.models.collaborative.svd import SVDModel
from src.evaluation.evaluator import Evaluator
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.result_manager import ResultManager


def print_section(title):
    """Imprime um cabeçalho de seção."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def main():
    """Função principal de exemplo."""
    
    # Configuração inicial
    print_section("SISTEMA DE RECOMENDAÇÃO - EXEMPLO DE USO")
    
    config = Config()
    logger = Logger('example')
    result_manager = ResultManager()
    
    # 1. Carregamento de dados
    print_section("1. CARREGANDO DADOS")
    
    loader = MovieLens100KLoader()
    data = loader.load()
    items_data = loader.load_items()
    metadata = loader.get_metadata()
    
    print(f"Dataset: {metadata['name']}")
    print(f"Usuários: {metadata['n_users']:,}")
    print(f"Itens: {metadata['n_items']:,}")
    print(f"Avaliações: {metadata['n_ratings']:,}")
    print(f"Esparsidade: {metadata['sparsity']*100:.2f}%")
    
    # 2. Pré-processamento
    print_section("2. PRÉ-PROCESSAMENTO")
    
    preprocessor = DataPreprocessor(min_user_ratings=5, min_item_ratings=5)
    data_filtered = preprocessor.filter_data(data)
    
    stats = preprocessor.get_statistics(data_filtered)
    print(f"Após filtragem:")
    print(f"- Avaliações: {stats['n_ratings']:,}")
    print(f"- Média de avaliações por usuário: {stats['avg_ratings_per_user']:.1f}")
    print(f"- Média de avaliações por item: {stats['avg_ratings_per_item']:.1f}")
    
    # 3. Divisão treino/teste
    print_section("3. DIVISÃO TREINO/TESTE")
    
    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_data, test_data = splitter.random_split(data_filtered)
    
    split_info = splitter.get_split_info(train_data, test_data)
    print(f"Treino: {split_info['train_size']:,} avaliações")
    print(f"Teste: {split_info['test_size']:,} avaliações")
    print(f"Usuários em comum: {split_info['common_users']:,}")
    print(f"Itens em comum: {split_info['common_items']:,}")
    
    # 4. Treinamento de modelos
    print_section("4. TREINANDO MODELOS")
    
    models = {
        'global_mean': GlobalMeanModel(),
        'popularity': PopularityModel(popularity_metric='mean_rating'),
        'knn_user': KNNUserModel(k=20, sim_metric='cosine'),
        'knn_item': KNNItemModel(k=20, sim_metric='cosine'),
        'svd': SVDModel(n_factors=50, n_epochs=10)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        model.fit(train_data)
        trained_models[name] = model
    
    # 5. Avaliação
    print_section("5. AVALIANDO MODELOS")
    
    evaluator = Evaluator()
    results = {}
    
    for name, model in trained_models.items():
        print(f"\nAvaliando {name}...")
        result = evaluator.evaluate_model(model, train_data, test_data)
        results[name] = result
    
    # 6. Comparação de resultados
    print_section("6. COMPARAÇÃO DE RESULTADOS")
    
    # Cria DataFrame comparativo
    comparison_data = []
    for model_name, result in results.items():
        row = {
            'Modelo': model_name,
            'RMSE': f"{result.get('rmse', 0):.3f}",
            'MAE': f"{result.get('mae', 0):.3f}",
            'Tempo Treino': f"{result.get('training_time', 0):.2f}s",
            'Cobertura': f"{result.get('coverage', 0)*100:.1f}%"
        }
        
        # Adiciona Precision@10 se disponível
        if 'ranking' in result and 'at_10' in result['ranking']:
            row['Precision@10'] = f"{result['ranking']['at_10'].get('precision', 0):.3f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nTabela Comparativa:")
    print(comparison_df.to_string(index=False))
    
    # 7. Exemplo de recomendação
    print_section("7. EXEMPLO DE RECOMENDAÇÃO")
    
    # Seleciona um usuário aleatório
    user_id = train_data['user_id'].sample(1).values[0]
    print(f"\nGerando recomendações para usuário {user_id}:")
    
    # Histórico do usuário
    user_history = train_data[train_data['user_id'] == user_id].copy()
    user_history = user_history.merge(items_data[['item_id', 'title']], on='item_id', how='left')
    
    print(f"\nHistórico do usuário (últimas 5 avaliações):")
    print(user_history[['title', 'rating']].tail(5).to_string(index=False))
    
    # Recomendações de cada modelo
    print("\nRecomendações:")
    for name, model in trained_models.items():
        print(f"\n{name.upper()}:")
        recommendations = model.recommend(user_id, n_items=5)
        
        for i, (item_id, score) in enumerate(recommendations, 1):
            title = items_data[items_data['item_id'] == item_id]['title'].values[0]
            print(f"  {i}. {title[:50]:<50} (score: {score:.3f})")
    
    # 8. Salvando resultados
    print_section("8. SALVANDO RESULTADOS")
    
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for model_name, result in results.items():
        path = result_manager.save_experiment_results(
            experiment_id=f"example_{experiment_id}",
            model_name=model_name,
            dataset_name='movielens-100k',
            metrics=result,
            model_params=trained_models[model_name].params
        )
        print(f"Resultados de {model_name} salvos em: {path}")
    
    print("\n" + "="*60)
    print(" EXEMPLO CONCLUÍDO!")
    print("="*60)
    print("\nPróximos passos:")
    print("1. Execute o dashboard: streamlit run src/dashboard/app.py")
    print("2. Explore os resultados salvos em: data/results/")
    print("3. Adicione novos modelos em: src/models/")
    print("4. Execute experimentos completos: python experiments/run_experiments.py")


if __name__ == "__main__":
    main()