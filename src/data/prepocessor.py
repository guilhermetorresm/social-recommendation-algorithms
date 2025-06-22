# src/data/preprocessor.py

class DataPreprocessor:
    """Pré-processador de dados para sistemas de recomendação."""
    
    def __init__(self, min_user_ratings: int = 5, min_item_ratings: int = 5):
        """
        Inicializa o pré-processador.
        
        Args:
            min_user_ratings: Mínimo de avaliações por usuário
            min_item_ratings: Mínimo de avaliações por item
        """
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        
    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra dados removendo usuários e itens com poucas avaliações.
        
        Args:
            data: DataFrame com colunas ['user_id', 'item_id', 'rating']
            
        Returns:
            pd.DataFrame: Dados filtrados
        """
        # Contagem de avaliações
        user_counts = data['user_id'].value_counts()
        item_counts = data['item_id'].value_counts()
        
        # Filtra usuários
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        data_filtered = data[data['user_id'].isin(valid_users)]
        
        # Filtra itens
        valid_items = item_counts[item_counts >= self.min_item_ratings].index
        data_filtered = data_filtered[data_filtered['item_id'].isin(valid_items)]
        
        print(f"Dados originais: {len(data)} avaliações")
        print(f"Dados filtrados: {len(data_filtered)} avaliações")
        print(f"Redução: {(1 - len(data_filtered)/len(data))*100:.2f}%")
        
        return data_filtered
    
    def normalize_ratings(self, data: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Normaliza as avaliações.
        
        Args:
            data: DataFrame com avaliações
            method: Método de normalização ('mean', 'zscore')
            
        Returns:
            pd.DataFrame: Dados normalizados
        """
        data_norm = data.copy()
        
        if method == 'mean':
            # Normalização pela média do usuário
            user_means = data.groupby('user_id')['rating'].mean()
            data_norm['rating_normalized'] = data_norm.apply(
                lambda x: x['rating'] - user_means[x['user_id']], 
                axis=1
            )
        elif method == 'zscore':
            # Z-score por usuário
            user_stats = data.groupby('user_id')['rating'].agg(['mean', 'std'])
            data_norm['rating_normalized'] = data_norm.apply(
                lambda x: (x['rating'] - user_stats.loc[x['user_id'], 'mean']) / 
                          (user_stats.loc[x['user_id'], 'std'] + 1e-8),
                axis=1
            )
        
        return data_norm
    
    def create_interaction_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria matriz de interações usuário-item.
        
        Args:
            data: DataFrame com avaliações
            
        Returns:
            pd.DataFrame: Matriz usuário-item
        """
        return pd.pivot_table(
            data,
            values='rating',
            index='user_id',
            columns='item_id',
            fill_value=0
        )
    
    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas do dataset."""
        stats = {
            'n_ratings': len(data),
            'n_users': data['user_id'].nunique(),
            'n_items': data['item_id'].nunique(),
            'avg_ratings_per_user': len(data) / data['user_id'].nunique(),
            'avg_ratings_per_item': len(data) / data['item_id'].nunique(),
            'rating_mean': data['rating'].mean(),
            'rating_std': data['rating'].std(),
            'sparsity': 1 - (len(data) / (data['user_id'].nunique() * data['item_id'].nunique()))
        }
        
        return stats

