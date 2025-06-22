# src/data/data_loader.py

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import requests
import zipfile
from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Classe abstrata para carregadores de dataset."""
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carrega o dataset."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados do dataset."""
        pass


class MovieLens100KLoader(DatasetLoader):
    """Carregador para o dataset MovieLens 100K."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        self.dataset_name = 'ml-100k'
        self.url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        
    def download_if_needed(self) -> str:
        """Baixa o dataset se necessário."""
        dataset_path = os.path.join(self.data_dir, self.dataset_name)
        
        if not os.path.exists(dataset_path):
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Download
            print(f"Baixando {self.dataset_name}...")
            response = requests.get(self.url)
            zip_path = os.path.join(self.data_dir, f'{self.dataset_name}.zip')
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extrai
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Remove o zip
            os.remove(zip_path)
            print(f"Dataset {self.dataset_name} baixado com sucesso!")
        
        return dataset_path
    
    def load(self) -> pd.DataFrame:
        """Carrega o dataset MovieLens 100K."""
        dataset_path = self.download_if_needed()
        
        # Carrega ratings
        ratings_path = os.path.join(dataset_path, 'u.data')
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        return ratings
    
    def load_items(self) -> pd.DataFrame:
        """Carrega informações dos itens (filmes)."""
        dataset_path = self.download_if_needed()
        
        # Carrega informações dos filmes
        items_path = os.path.join(dataset_path, 'u.item')
        
        # Gêneros
        genres = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Colunas
        item_columns = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        item_columns.extend(genres)
        
        items = pd.read_csv(
            items_path,
            sep='|',
            names=item_columns,
            encoding='latin-1'
        )
        
        return items
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados do dataset."""
        ratings = self.load()
        
        return {
            'name': 'MovieLens 100K',
            'n_users': ratings['user_id'].nunique(),
            'n_items': ratings['item_id'].nunique(),
            'n_ratings': len(ratings),
            'rating_scale': (ratings['rating'].min(), ratings['rating'].max()),
            'sparsity': 1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique())),
            'rating_distribution': ratings['rating'].value_counts().to_dict()
        }
