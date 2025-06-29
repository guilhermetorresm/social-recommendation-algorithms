#!/usr/bin/env python3
# example_deep_learning.py

"""
Exemplo de uso dos modelos de Deep Learning implementados.
Este script demonstra como usar os modelos Two-Tower e NCF.
"""

import sys
import os
import pandas as pd
import numpy as np

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import MovieLens100KLoader
from data.preprocessor import DataPreprocessor
from data.splitter import DataSplitter
from models.deep_learning.two_tower import TwoTowerModel
from models.deep_learning.ncf import NCFModel
from evaluation.evaluator import Evaluator


def exemplo_basico():
    """Exemplo básico de uso dos modelos de Deep Learning."""
    print("🚀 Exemplo de Deep Learning para Sistemas de Recomendação")
    print("="*60)
    
    # 1. Carrega dados (apenas uma pequena amostra para exemplo rápido)
    print("📊 Carregando dados MovieLens 100K...")
    loader = MovieLens100KLoader('data/raw')
    data = loader.load()
    
    # Para exemplo rápido, usa apenas uma amostra
    print(f"Dataset original: {len(data)} interações")
    data_sample = data.sample(n=min(5000, len(data)), random_state=42)
    print(f"Amostra para exemplo: {len(data_sample)} interações")
    
    # 2. Pré-processa e divide dados
    print("\n🔧 Pré-processando dados...")
    preprocessor = DataPreprocessor(min_user_ratings=3, min_item_ratings=3)
    data_filtered = preprocessor.filter_data(data_sample)
    
    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_data, test_data = splitter.random_split(data_filtered)
    
    print(f"Dados de treino: {len(train_data)} interações")
    print(f"Dados de teste: {len(test_data)} interações")
    
    # 3. Configura modelos com parâmetros reduzidos para exemplo rápido
    print("\n🧠 Configurando modelos de Deep Learning...")
    
    # Two-Tower com parâmetros reduzidos
    two_tower_params = {
        'user_embedding_dim': 32,
        'item_embedding_dim': 32,
        'user_tower_units': [64, 32],
        'item_tower_units': [64, 32],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 5,  # Reduzido para exemplo rápido
        'batch_size': 128,
        'early_stopping_patience': 3
    }
    
    # NCF com parâmetros reduzidos
    ncf_params = {
        'mf_embedding_dim': 16,
        'mlp_embedding_dim': 16,
        'hidden_units': [64, 32, 16],
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 5,  # Reduzido para exemplo rápido
        'batch_size': 128,
        'early_stopping_patience': 3
    }
    
    # 4. Treina modelos
    print("\n🏋️‍♂️ Treinando modelos...")
    
    # Two-Tower
    print("\n" + "-"*40)
    print("🏗️  Treinando Two-Tower Model...")
    print("-"*40)
    two_tower = TwoTowerModel(**two_tower_params)
    two_tower.fit(train_data)
    
    # NCF
    print("\n" + "-"*40)
    print("🧩 Treinando NCF Model...")
    print("-"*40)
    ncf = NCFModel(**ncf_params)
    ncf.fit(train_data)
    
    # 5. Faz predições
    print("\n🔮 Testando predições...")
    
    # Pega um usuário e item de exemplo
    sample_user = train_data['user_id'].iloc[0]
    sample_item = train_data['item_id'].iloc[0]
    
    tt_prediction = two_tower.predict(sample_user, sample_item)
    ncf_prediction = ncf.predict(sample_user, sample_item)
    
    print(f"Usuário: {sample_user}, Item: {sample_item}")
    print(f"Two-Tower predição: {tt_prediction:.3f}")
    print(f"NCF predição: {ncf_prediction:.3f}")
    
    # 6. Gera recomendações
    print(f"\n📝 Gerando recomendações para usuário {sample_user}...")
    
    tt_recs = two_tower.recommend(sample_user, n_items=5)
    ncf_recs = ncf.recommend(sample_user, n_items=5)
    
    print("\nTwo-Tower Top-5 recomendações:")
    for i, (item_id, score) in enumerate(tt_recs, 1):
        print(f"  {i}. Item {item_id}: {score:.3f}")
    
    print("\nNCF Top-5 recomendações:")
    for i, (item_id, score) in enumerate(ncf_recs, 1):
        print(f"  {i}. Item {item_id}: {score:.3f}")
    
    # 7. Análise de embeddings (Two-Tower)
    print(f"\n🔍 Analisando representações aprendidas...")
    
    user_repr = two_tower.get_user_representation(sample_user)
    item_repr = two_tower.get_item_representation(sample_item)
    
    print(f"Representação usuário {sample_user}: shape {user_repr.shape}")
    print(f"Representação item {sample_item}: shape {item_repr.shape}")
    
    # Usuários similares
    similar_users = two_tower.get_similar_users(sample_user, n_users=3)
    print(f"\nUsuários similares ao usuário {sample_user}:")
    for user_id, similarity in similar_users:
        print(f"  Usuário {user_id}: {similarity:.3f}")
    
    # 8. Análise NCF
    print(f"\n🔬 Analisando componentes NCF...")
    
    ncf_analysis = ncf.analyze_model_components(sample_user, sample_item)
    print(f"Análise NCF para usuário {sample_user}, item {sample_item}:")
    for key, value in ncf_analysis.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Exemplo concluído com sucesso!")
    print("\n💡 Para experimentos completos, use:")
    print("   python experiments/run_experiments.py --mode deep_learning")


def exemplo_avaliacao_rapida():
    """Exemplo de avaliação rápida dos modelos."""
    print("\n" + "="*60)
    print("📊 Avaliação Rápida dos Modelos de Deep Learning")
    print("="*60)
    
    # Carrega dados pequenos
    loader = MovieLens100KLoader('data/raw')
    data = loader.load().sample(n=1000, random_state=42)
    
    # Pré-processa
    preprocessor = DataPreprocessor(min_user_ratings=2, min_item_ratings=2)
    data_filtered = preprocessor.filter_data(data)
    
    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_data, test_data = splitter.random_split(data_filtered)
    
    # Modelo simples para avaliação rápida
    print("🚀 Treinando modelo NCF simples...")
    ncf = NCFModel(
        mf_embedding_dim=8,
        mlp_embedding_dim=8,
        hidden_units=[32, 16],
        epochs=3,
        batch_size=64
    )
    ncf.fit(train_data)
    
    # Avaliação básica
    print("📏 Avaliando modelo...")
    evaluator = Evaluator()
    
    # Teste de predição simples
    predictions = []
    true_ratings = []
    
    for _, row in test_data.head(50).iterrows():  # Apenas 50 para ser rápido
        pred = ncf.predict(row['user_id'], row['item_id'])
        predictions.append(pred)
        true_ratings.append(row['rating'])
    
    # Calcula RMSE simples
    rmse = np.sqrt(np.mean([(p - t)**2 for p, t in zip(predictions, true_ratings)]))
    mae = np.mean([abs(p - t) for p, t in zip(predictions, true_ratings)])
    
    print(f"✅ Resultados da avaliação rápida:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Predições testadas: {len(predictions)}")


if __name__ == "__main__":
    # Verifica se TensorFlow está disponível
    try:
        import tensorflow as tf
        print(f"TensorFlow versão: {tf.__version__}")
        print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("⚠️  TensorFlow não encontrado. Instale com: pip install tensorflow")
        sys.exit(1)
    
    try:
        exemplo_basico()
        exemplo_avaliacao_rapida()
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Dicas para resolver problemas:")
        print("   1. Certifique-se que o dataset MovieLens foi baixado")
        print("   2. Verifique se todas as dependências estão instaladas")
        print("   3. Certifique-se que o TensorFlow está funcionando")
        print("   4. Execute: python -c 'import tensorflow; print(tensorflow.__version__)'") 