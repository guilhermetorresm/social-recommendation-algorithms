# src/models/deep_learning/two_tower.py

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import time
import warnings
from ..base_model import BaseRecommender


class TwoTowerModel(BaseRecommender):
    """
    Two-Tower Architecture para sistemas de recomendação.
    
    Este modelo utiliza duas torres separadas (user tower e item tower) para
    aprender representações latentes de usuários e itens, que são então
    combinadas para fazer predições de rating.
    """
    
    def __init__(self, 
                 user_embedding_dim: int = 64,
                 item_embedding_dim: int = 64,
                 user_tower_units: List[int] = [128, 64, 32],
                 item_tower_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 256,
                 early_stopping_patience: int = 10,
                 l2_reg: float = 0.001,
                 **kwargs):
        """
        Inicializa o modelo Two-Tower.
        
        Args:
            user_embedding_dim: Dimensão dos embeddings de usuário
            item_embedding_dim: Dimensão dos embeddings de item
            user_tower_units: Lista com tamanhos das camadas da torre de usuário
            item_tower_units: Lista com tamanhos das camadas da torre de item
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
            epochs: Número de épocas
            batch_size: Tamanho do batch
            early_stopping_patience: Paciência para early stopping
            l2_reg: Regularização L2
        """
        super().__init__(
            model_name="TwoTower",
            model_type="deep_learning",
            user_embedding_dim=user_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            user_tower_units=user_tower_units,
            item_tower_units=item_tower_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            l2_reg=l2_reg,
            **kwargs
        )
        
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.user_tower_units = user_tower_units
        self.item_tower_units = item_tower_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        
        # Componentes do modelo
        self.model = None
        self.user_tower_model = None
        self.item_tower_model = None
        
        # Mapeamentos e estruturas auxiliares
        self.user_encoder = None
        self.item_encoder = None
        self.n_users = None
        self.n_items = None
        self.user_items = {}
        self.item_users = {}
        self.rating_scale = None
        self.global_mean = None
    
    def _build_user_tower(self) -> tf.keras.Model:
        """Constrói a torre de usuário."""
        user_input = tf.keras.Input(shape=(), name='user_id')
        
        # Embedding de usuário
        user_embedding = tf.keras.layers.Embedding(
            self.n_users, 
            self.user_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='user_embedding'
        )(user_input)
        
        # Flatten
        x = tf.keras.layers.Flatten()(user_embedding)
        
        # Camadas densas da torre de usuário
        for i, units in enumerate(self.user_tower_units):
            x = tf.keras.layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name=f'user_dense_{i}'
            )(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Camada de saída da torre (representação final do usuário)
        user_representation = tf.keras.layers.Dense(
            32, 
            activation='relu',
            name='user_output'
        )(x)
        
        return tf.keras.Model(inputs=user_input, outputs=user_representation, name='user_tower')
    
    def _build_item_tower(self) -> tf.keras.Model:
        """Constrói a torre de item."""
        item_input = tf.keras.Input(shape=(), name='item_id')
        
        # Embedding de item
        item_embedding = tf.keras.layers.Embedding(
            self.n_items, 
            self.item_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='item_embedding'
        )(item_input)
        
        # Flatten
        x = tf.keras.layers.Flatten()(item_embedding)
        
        # Camadas densas da torre de item
        for i, units in enumerate(self.item_tower_units):
            x = tf.keras.layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name=f'item_dense_{i}'
            )(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Camada de saída da torre (representação final do item)
        item_representation = tf.keras.layers.Dense(
            32, 
            activation='relu',
            name='item_output'
        )(x)
        
        return tf.keras.Model(inputs=item_input, outputs=item_representation, name='item_tower')
    
    def _build_interaction_model(self) -> tf.keras.Model:
        """Constrói o modelo completo combinando as duas torres."""
        user_input = tf.keras.Input(shape=(), name='user_id')
        item_input = tf.keras.Input(shape=(), name='item_id')
        
        # Obter representações das torres
        user_repr = self.user_tower_model(user_input)
        item_repr = self.item_tower_model(item_input)
        
        # Diferentes formas de combinar as representações
        # 1. Produto escalar (dot product)
        dot_product = tf.keras.layers.Dot(axes=1)([user_repr, item_repr])
        
        # 2. Concatenação + camadas densas
        concat = tf.keras.layers.Concatenate()([user_repr, item_repr])
        
        # Camadas de interação
        x = tf.keras.layers.Dense(64, activation='relu')(concat)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Combina dot product com features de interação
        combined = tf.keras.layers.Concatenate()([dot_product, x])
        
        # Camada final para predição de rating
        output = tf.keras.layers.Dense(1, activation='linear', name='rating_prediction')(combined)
        
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        return model
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'TwoTowerModel':
        """
        Treina o modelo Two-Tower.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        print(f"Parâmetros: user_emb_dim={self.user_embedding_dim}, item_emb_dim={self.item_embedding_dim}")
        start_time = time.time()
        
        # Prepara mapeamentos de usuários e itens
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['item_id'].unique())
        
        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.item_encoder = {item: idx for idx, item in enumerate(unique_items)}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        self.rating_scale = (train_data['rating'].min(), train_data['rating'].max())
        self.global_mean = train_data['rating'].mean()
        
        # Estruturas auxiliares para recomendação
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
        
        # Prepara dados de treino
        user_ids = train_data['user_id'].map(self.user_encoder).values
        item_ids = train_data['item_id'].map(self.item_encoder).values
        ratings = train_data['rating'].values
        
        # Constrói torres
        print("Construindo torres...")
        self.user_tower_model = self._build_user_tower()
        self.item_tower_model = self._build_item_tower()
        
        # Constrói modelo completo
        self.model = self._build_interaction_model()
        
        # Compila modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Treina modelo
        print(f"Iniciando treinamento com {len(train_data)} interações...")
        history = self.model.fit(
            [user_ids, item_ids], 
            ratings,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"Modelo treinado em {self.training_time:.2f}s")
        print(f"Torres treinadas para {self.n_users} usuários e {self.n_items} itens")
        print(f"Melhor val_loss: {min(history.history['val_loss']):.4f}")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz o rating para um par usuário-item.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            float: Rating predito
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        # Verifica se usuário e item existem nos dados de treino
        if user_id not in self.user_encoder or item_id not in self.item_encoder:
            return self.global_mean
        
        # Codifica usuário e item
        user_idx = self.user_encoder[user_id]
        item_idx = self.item_encoder[item_id]
        
        # Faz predição
        prediction = self.model.predict(
            [np.array([user_idx]), np.array([item_idx])],
            verbose=0
        )[0][0]
        
        # Clipa para escala válida
        min_rating, max_rating = self.rating_scale
        return np.clip(prediction, min_rating, max_rating)
    
    def recommend(self, user_id: int, n_items: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Recomenda top-N itens para um usuário usando as torres.
        
        Args:
            user_id: ID do usuário
            n_items: Número de itens a recomendar
            exclude_seen: Se deve excluir itens já vistos
            
        Returns:
            List[Tuple[int, float]]: Lista de tuplas (item_id, score)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        if user_id not in self.user_encoder:
            return []  # Usuário não conhecido
        
        # Itens já vistos pelo usuário
        seen_items = set(self.user_items.get(user_id, {}).keys())
        
        # Todos os itens disponíveis
        all_items = set(self.item_encoder.keys())
        
        # Candidatos para recomendação
        if exclude_seen:
            candidate_items = all_items - seen_items
        else:
            candidate_items = all_items
        
        if not candidate_items:
            return []
        
        # Prepara dados para predição em batch
        user_idx = self.user_encoder[user_id]
        candidate_items = list(candidate_items)
        item_indices = [self.item_encoder[item] for item in candidate_items]
        
        user_indices = [user_idx] * len(candidate_items)
        
        # Faz predições em batch
        predictions = self.model.predict(
            [np.array(user_indices), np.array(item_indices)],
            verbose=0
        ).flatten()
        
        # Combina itens com suas predições
        item_scores = list(zip(candidate_items, predictions))
        
        # Ordena por score decrescente e retorna top-N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n_items]
    
    def get_user_representation(self, user_id: int) -> np.ndarray:
        """
        Obtém a representação aprendida de um usuário da torre de usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            np.ndarray: Representação do usuário
        """
        if not self.is_fitted or user_id not in self.user_encoder:
            return np.zeros(32)  # Tamanho da representação final
        
        user_idx = self.user_encoder[user_id]
        user_repr = self.user_tower_model.predict(
            np.array([user_idx]), verbose=0
        )[0]
        
        return user_repr
    
    def get_item_representation(self, item_id: int) -> np.ndarray:
        """
        Obtém a representação aprendida de um item da torre de item.
        
        Args:
            item_id: ID do item
            
        Returns:
            np.ndarray: Representação do item
        """
        if not self.is_fitted or item_id not in self.item_encoder:
            return np.zeros(32)  # Tamanho da representação final
        
        item_idx = self.item_encoder[item_id]
        item_repr = self.item_tower_model.predict(
            np.array([item_idx]), verbose=0
        )[0]
        
        return item_repr
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        """
        Encontra usuários similares baseado nas representações da torre de usuário.
        
        Args:
            user_id: ID do usuário
            n_users: Número de usuários similares a retornar
            
        Returns:
            List[Tuple[int, float]]: Lista de (user_id, similarity_score)
        """
        if not self.is_fitted or user_id not in self.user_encoder:
            return []
        
        user_repr = self.get_user_representation(user_id)
        
        similarities = []
        for other_user_id in self.user_encoder.keys():
            if other_user_id != user_id:
                other_repr = self.get_user_representation(other_user_id)
                # Similaridade cosseno
                similarity = np.dot(user_repr, other_repr) / (
                    np.linalg.norm(user_repr) * np.linalg.norm(other_repr) + 1e-8
                )
                similarities.append((other_user_id, similarity))
        
        # Ordena por similaridade decrescente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_users]
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Encontra itens similares baseado nas representações da torre de item.
        
        Args:
            item_id: ID do item
            n_items: Número de itens similares a retornar
            
        Returns:
            List[Tuple[int, float]]: Lista de (item_id, similarity_score)
        """
        if not self.is_fitted or item_id not in self.item_encoder:
            return []
        
        item_repr = self.get_item_representation(item_id)
        
        similarities = []
        for other_item_id in self.item_encoder.keys():
            if other_item_id != item_id:
                other_repr = self.get_item_representation(other_item_id)
                # Similaridade cosseno
                similarity = np.dot(item_repr, other_repr) / (
                    np.linalg.norm(item_repr) * np.linalg.norm(other_repr) + 1e-8
                )
                similarities.append((other_item_id, similarity))
        
        # Ordena por similaridade decrescente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_items] 