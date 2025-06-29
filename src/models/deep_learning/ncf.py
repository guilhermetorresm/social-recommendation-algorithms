# src/models/deep_learning/ncf.py

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import time
import warnings
from ..base_model import BaseRecommender


class NCFModel(BaseRecommender):
    """
    Neural Collaborative Filtering (NCF) para sistemas de recomendação.
    
    Este modelo combina Matrix Factorization com Multi-Layer Perceptrons,
    usando embeddings de usuários e itens que são processados através de
    redes neurais para capturar interações não-lineares complexas.
    """
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 mf_embedding_dim: int = 32,
                 mlp_embedding_dim: int = 32,
                 hidden_units: List[int] = [128, 64, 32, 16],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 256,
                 early_stopping_patience: int = 10,
                 l2_reg: float = 0.001,
                 use_bias: bool = True,
                 **kwargs):
        """
        Inicializa o modelo NCF.
        
        Args:
            embedding_dim: Dimensão geral dos embeddings (usado se não especificar MF/MLP separadamente)
            mf_embedding_dim: Dimensão dos embeddings para a parte Matrix Factorization
            mlp_embedding_dim: Dimensão dos embeddings para a parte MLP
            hidden_units: Lista com tamanhos das camadas ocultas do MLP
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
            epochs: Número de épocas
            batch_size: Tamanho do batch
            early_stopping_patience: Paciência para early stopping
            l2_reg: Regularização L2
            use_bias: Se deve usar bias nas camadas densas
        """
        super().__init__(
            model_name="NCF",
            model_type="deep_learning",
            embedding_dim=embedding_dim,
            mf_embedding_dim=mf_embedding_dim,
            mlp_embedding_dim=mlp_embedding_dim,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            l2_reg=l2_reg,
            use_bias=use_bias,
            **kwargs
        )
        
        self.embedding_dim = embedding_dim
        self.mf_embedding_dim = mf_embedding_dim or embedding_dim // 2
        self.mlp_embedding_dim = mlp_embedding_dim or embedding_dim // 2
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        
        # Componentes do modelo
        self.model = None
        
        # Mapeamentos e estruturas auxiliares
        self.user_encoder = None
        self.item_encoder = None
        self.n_users = None
        self.n_items = None
        self.user_items = {}
        self.item_users = {}
        self.rating_scale = None
        self.global_mean = None
    
    def _build_model(self) -> tf.keras.Model:
        """Constrói a arquitetura NCF completa."""
        # Inputs
        user_input = tf.keras.Input(shape=(), name='user_id')
        item_input = tf.keras.Input(shape=(), name='item_id')
        
        # =============================================================
        # PARTE 1: Generalized Matrix Factorization (GMF)
        # =============================================================
        
        # Embeddings para MF
        mf_user_embedding = tf.keras.layers.Embedding(
            self.n_users, 
            self.mf_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='mf_user_embedding'
        )(user_input)
        
        mf_item_embedding = tf.keras.layers.Embedding(
            self.n_items, 
            self.mf_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='mf_item_embedding'
        )(item_input)
        
        # Flatten embeddings
        mf_user_vec = tf.keras.layers.Flatten()(mf_user_embedding)
        mf_item_vec = tf.keras.layers.Flatten()(mf_item_embedding)
        
        # Element-wise product (Hadamard product)
        mf_output = tf.keras.layers.Multiply(name='mf_multiply')([mf_user_vec, mf_item_vec])
        
        # =============================================================
        # PARTE 2: Multi-Layer Perceptron (MLP)
        # =============================================================
        
        # Embeddings separados para MLP
        mlp_user_embedding = tf.keras.layers.Embedding(
            self.n_users, 
            self.mlp_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='mlp_user_embedding'
        )(user_input)
        
        mlp_item_embedding = tf.keras.layers.Embedding(
            self.n_items, 
            self.mlp_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='mlp_item_embedding'
        )(item_input)
        
        # Flatten embeddings
        mlp_user_vec = tf.keras.layers.Flatten()(mlp_user_embedding)
        mlp_item_vec = tf.keras.layers.Flatten()(mlp_item_embedding)
        
        # Concatenate user and item embeddings
        mlp_concat = tf.keras.layers.Concatenate(name='mlp_concat')([mlp_user_vec, mlp_item_vec])
        
        # MLP layers
        mlp_output = mlp_concat
        for i, units in enumerate(self.hidden_units):
            mlp_output = tf.keras.layers.Dense(
                units, 
                activation='relu',
                use_bias=self.use_bias,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                name=f'mlp_dense_{i}'
            )(mlp_output)
            mlp_output = tf.keras.layers.Dropout(self.dropout_rate)(mlp_output)
        
        # =============================================================
        # PARTE 3: Fusion Layer - Combina GMF e MLP
        # =============================================================
        
        # Concatena saídas do MF e MLP
        fusion = tf.keras.layers.Concatenate(name='fusion')([mf_output, mlp_output])
        
        # Camada final para predição
        output = tf.keras.layers.Dense(
            1, 
            activation='linear',
            use_bias=self.use_bias,
            kernel_initializer='lecun_uniform',
            name='rating_prediction'
        )(fusion)
        
        # Cria modelo
        model = tf.keras.Model(
            inputs=[user_input, item_input], 
            outputs=output, 
            name='NCF'
        )
        
        return model
    
    def fit(self, train_data: pd.DataFrame, **kwargs) -> 'NCFModel':
        """
        Treina o modelo NCF.
        
        Args:
            train_data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            self: Instância do modelo treinado
        """
        print(f"Treinando {self.model_name}...")
        print(f"Parâmetros: mf_emb={self.mf_embedding_dim}, mlp_emb={self.mlp_embedding_dim}")
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
        
        # Constrói modelo
        print("Construindo modelo NCF...")
        self.model = self._build_model()
        
        # Compila modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        # Exibe arquitetura
        print("\nArquitetura do modelo NCF:")
        self.model.summary()
        
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
        print(f"Iniciando treinamento NCF com {len(train_data)} interações...")
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
        
        print(f"Modelo NCF treinado em {self.training_time:.2f}s")
        print(f"Embeddings aprendidos para {self.n_users} usuários e {self.n_items} itens")
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
        Recomenda top-N itens para um usuário.
        
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
    
    def get_user_mf_embedding(self, user_id: int) -> np.ndarray:
        """
        Obtém o embedding MF de um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            np.ndarray: Embedding MF do usuário
        """
        if not self.is_fitted or user_id not in self.user_encoder:
            return np.zeros(self.mf_embedding_dim)
        
        user_idx = self.user_encoder[user_id]
        mf_user_layer = self.model.get_layer('mf_user_embedding')
        return mf_user_layer.get_weights()[0][user_idx]
    
    def get_item_mf_embedding(self, item_id: int) -> np.ndarray:
        """
        Obtém o embedding MF de um item.
        
        Args:
            item_id: ID do item
            
        Returns:
            np.ndarray: Embedding MF do item
        """
        if not self.is_fitted or item_id not in self.item_encoder:
            return np.zeros(self.mf_embedding_dim)
        
        item_idx = self.item_encoder[item_id]
        mf_item_layer = self.model.get_layer('mf_item_embedding')
        return mf_item_layer.get_weights()[0][item_idx]
    
    def get_user_mlp_embedding(self, user_id: int) -> np.ndarray:
        """
        Obtém o embedding MLP de um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            np.ndarray: Embedding MLP do usuário
        """
        if not self.is_fitted or user_id not in self.user_encoder:
            return np.zeros(self.mlp_embedding_dim)
        
        user_idx = self.user_encoder[user_id]
        mlp_user_layer = self.model.get_layer('mlp_user_embedding')
        return mlp_user_layer.get_weights()[0][user_idx]
    
    def get_item_mlp_embedding(self, item_id: int) -> np.ndarray:
        """
        Obtém o embedding MLP de um item.
        
        Args:
            item_id: ID do item
            
        Returns:
            np.ndarray: Embedding MLP do item
        """
        if not self.is_fitted or item_id not in self.item_encoder:
            return np.zeros(self.mlp_embedding_dim)
        
        item_idx = self.item_encoder[item_id]
        mlp_item_layer = self.model.get_layer('mlp_item_embedding')
        return mlp_item_layer.get_weights()[0][item_idx]
    
    def analyze_model_components(self, user_id: int, item_id: int) -> Dict[str, float]:
        """
        Analisa as contribuições dos componentes MF e MLP para uma predição.
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
            
        Returns:
            Dict com análise dos componentes
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Use fit() primeiro.")
        
        if user_id not in self.user_encoder or item_id not in self.item_encoder:
            return {}
        
        user_idx = self.user_encoder[user_id]
        item_idx = self.item_encoder[item_id]
        
        # Cria modelos intermediários para extrair saídas dos componentes
        user_input = self.model.get_layer('user_id')
        item_input = self.model.get_layer('item_id')
        
        # Saída do componente MF
        mf_output = self.model.get_layer('mf_multiply')
        mf_model = tf.keras.Model(
            inputs=[user_input.input, item_input.input],
            outputs=mf_output.output
        )
        
        # Saída do último layer MLP antes da fusão
        mlp_layers = [layer for layer in self.model.layers if 'mlp_dense' in layer.name]
        if mlp_layers:
            mlp_output = mlp_layers[-1]
            mlp_model = tf.keras.Model(
                inputs=[user_input.input, item_input.input],
                outputs=mlp_output.output
            )
        
        # Calcula saídas
        mf_score = mf_model.predict([np.array([user_idx]), np.array([item_idx])], verbose=0)[0]
        if mlp_layers:
            mlp_score = mlp_model.predict([np.array([user_idx]), np.array([item_idx])], verbose=0)[0]
        else:
            mlp_score = np.array([0])
        
        # Predição final
        final_score = self.predict(user_id, item_id)
        
        return {
            'mf_contribution': float(np.mean(mf_score)),
            'mlp_contribution': float(np.mean(mlp_score)),
            'final_prediction': final_score,
            'user_mf_norm': float(np.linalg.norm(self.get_user_mf_embedding(user_id))),
            'item_mf_norm': float(np.linalg.norm(self.get_item_mf_embedding(item_id))),
            'user_mlp_norm': float(np.linalg.norm(self.get_user_mlp_embedding(user_id))),
            'item_mlp_norm': float(np.linalg.norm(self.get_item_mlp_embedding(item_id)))
        } 