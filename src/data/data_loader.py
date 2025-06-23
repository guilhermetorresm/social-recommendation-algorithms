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


class MovieLens1MLoader(DatasetLoader):
    """Carregador para o dataset MovieLens 1M."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = data_dir
        self.dataset_name = 'ml-1m'
        self.url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        
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
        """Carrega o dataset MovieLens 1M."""
        dataset_path = self.download_if_needed()
        
        # Carrega ratings (formato: UserID::MovieID::Rating::Timestamp)
        ratings_path = os.path.join(dataset_path, 'ratings.dat')
        ratings = pd.read_csv(
            ratings_path,
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'  # Necessário para separator de múltiplos caracteres
        )
        
        return ratings
    
    def load_users(self) -> pd.DataFrame:
        """Carrega informações dos usuários."""
        dataset_path = self.download_if_needed()
        
        # Carrega users (formato: UserID::Gender::Age::Occupation::Zip-code)
        users_path = os.path.join(dataset_path, 'users.dat')
        users = pd.read_csv(
            users_path,
            sep='::',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python'
        )
        
        return users
    
    def load_items(self) -> pd.DataFrame:
        """Carrega informações dos itens (filmes) - compatível com MovieLens100KLoader."""
        dataset_path = self.download_if_needed()
        
        # Carrega movies (formato: MovieID::Title::Genres)
        movies_path = os.path.join(dataset_path, 'movies.dat')
        movies = pd.read_csv(
            movies_path,
            sep='::',
            names=['item_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'
        )
        
        return movies
    
    def load_movies(self) -> pd.DataFrame:
        """Carrega informações dos filmes."""
        dataset_path = self.download_if_needed()
        
        # Carrega movies (formato: MovieID::Title::Genres)
        movies_path = os.path.join(dataset_path, 'movies.dat')
        movies = pd.read_csv(
            movies_path,
            sep='::',
            names=['item_id', 'title', 'genres'],
            engine='python',
            encoding='latin-1'  # Para caracteres especiais nos títulos
        )
        
        # Processa gêneros (separados por |)
        movies['genres_list'] = movies['genres'].str.split('|')
        
        return movies
    
    def load_complete(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carrega todos os dados (ratings, users, movies).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (ratings, users, movies)
        """
        return self.load(), self.load_users(), self.load_movies()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados do dataset."""
        ratings = self.load()
        users = self.load_users()
        movies = self.load_movies()
        
        return {
            'name': 'MovieLens 1M',
            'n_users': ratings['user_id'].nunique(),
            'n_items': ratings['item_id'].nunique(),
            'n_ratings': len(ratings),
            'rating_scale': (ratings['rating'].min(), ratings['rating'].max()),
            'sparsity': 1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique())),
            'rating_distribution': ratings['rating'].value_counts().to_dict(),
            'user_demographics': {
                'gender_distribution': users['gender'].value_counts().to_dict(),
                'age_distribution': users['age'].value_counts().to_dict(),
                'occupation_distribution': users['occupation'].value_counts().to_dict()
            },
            'movie_info': {
                'n_movies': len(movies),
                'genres': sorted(set([genre for genres_list in movies['genres_list'] 
                                    for genre in genres_list if genre != '(no genres listed)']))
            }
        }
    
    def get_genre_matrix(self) -> pd.DataFrame:
        """
        Cria matriz binária de gêneros por filme.
        
        Returns:
            pd.DataFrame: Matriz com filmes nas linhas e gêneros nas colunas
        """
        movies = self.load_movies()
        
        # Obtém todos os gêneros únicos
        all_genres = sorted(set([genre for genres_list in movies['genres_list'] 
                               for genre in genres_list if genre != '(no genres listed)']))
        
        # Cria matriz binária
        genre_matrix = pd.DataFrame(0, index=movies['item_id'], columns=all_genres)
        
        for idx, row in movies.iterrows():
            for genre in row['genres_list']:
                if genre != '(no genres listed)':
                    genre_matrix.loc[row['item_id'], genre] = 1
        
        return genre_matrix