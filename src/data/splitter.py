# src/data/splitter.py

from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataSplitter:
    """Divisor de dados para treino e teste."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Inicializa o divisor.
        
        Args:
            test_size: Proporção dos dados para teste
            random_state: Semente aleatória
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def random_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divisão aleatória dos dados.
        
        Args:
            data: DataFrame com avaliações
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (treino, teste)
        """
        train, test = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        return train, test
    
    def temporal_split(self, data: pd.DataFrame, 
                      timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divisão temporal dos dados.
        
        Args:
            data: DataFrame com avaliações
            timestamp_col: Nome da coluna de timestamp
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (treino, teste)
        """
        # Ordena por timestamp
        data_sorted = data.sort_values(timestamp_col)
        
        # Calcula o ponto de corte
        split_idx = int(len(data_sorted) * (1 - self.test_size))
        
        train = data_sorted.iloc[:split_idx]
        test = data_sorted.iloc[split_idx:]
        
        return train, test
    
    def user_based_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divisão baseada em usuários (deixa alguns ratings de cada usuário para teste).
        
        Args:
            data: DataFrame com avaliações
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (treino, teste)
        """
        train_list = []
        test_list = []
        
        # Para cada usuário
        for user_id, user_data in data.groupby('user_id'):
            # Garante pelo menos 1 item no treino e 1 no teste
            if len(user_data) < 2:
                train_list.append(user_data)
                continue
            
            # Divide os dados do usuário
            user_train, user_test = train_test_split(
                user_data,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            train_list.append(user_train)
            test_list.append(user_test)
        
        train = pd.concat(train_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)
        
        return train, test
    
    def get_split_info(self, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
        """Retorna informações sobre a divisão."""
        info = {
            'train_size': len(train),
            'test_size': len(test),
            'train_ratio': len(train) / (len(train) + len(test)),
            'train_users': train['user_id'].nunique(),
            'test_users': test['user_id'].nunique(),
            'train_items': train['item_id'].nunique(),
            'test_items': test['item_id'].nunique(),
            'common_users': len(set(train['user_id']) & set(test['user_id'])),
            'common_items': len(set(train['item_id']) & set(test['item_id']))
        }
        
        return info
