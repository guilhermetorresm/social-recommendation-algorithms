# 🎯 Social Recommendation Algorithms

Um framework completo e extensível para sistemas de recomendação, implementando algoritmos clássicos e modernos de machine learning e deep learning.

## 🚀 Características Principais

- **Arquitetura Modular**: Framework bem estruturado seguindo padrões de design limpos
- **Múltiplos Algoritmos**: Baseline, colaborativos, baseados em conteúdo e deep learning
- **Avaliação Abrangente**: Métricas de acurácia, ranking e beyond-accuracy
- **Configuração Flexível**: Sistema de configuração via YAML
- **Deep Learning Nativo**: Modelos Two-Tower e NCF implementados com TensorFlow
- **Experimentos Reproduzíveis**: Sistema completo de logging e resultados

## 📊 Modelos Implementados

### 🏗️ Baseline
- **Global Mean**: Média global de ratings
- **Popularity**: Baseado em popularidade (contagem ou média de ratings)

### 🤝 Colaborativos
- **KNN User-Based**: Filtragem colaborativa baseada em usuários
- **KNN Item-Based**: Filtragem colaborativa baseada em itens  
- **SVD**: Singular Value Decomposition usando Surprise

### 📝 Baseado em Conteúdo
- **Content-Based**: Recomendações baseadas em similaridade de gêneros

### 🧠 Deep Learning
- **Two-Tower**: Arquitetura de duas torres para embeddings separados de usuários e itens
- **NCF**: Neural Collaborative Filtering combinando Matrix Factorization e MLPs

## 🛠️ Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/social-recommendation-algorithms.git
cd social-recommendation-algorithms

# Instale as dependências
pip install -e .

# Ou com uv (recomendado)
uv sync
```

## 📈 Uso Rápido

### Experimentos Completos

```bash
# Executar apenas modelos de Deep Learning
python experiments/run_experiments.py --mode deep_learning

# Executar todos os modelos baseline
python experiments/run_experiments.py --mode baselines

# Executar todos os modelos
python experiments/run_experiments.py --mode all

# Exploração de KNN com diferentes parâmetros
python experiments/run_experiments.py --mode knn_explorer

# Apenas Content-Based
python experiments/run_experiments.py --mode content_based
```

### Exemplo de Uso Programático

```python
from src.models.deep_learning.two_tower import TwoTowerModel
from src.models.deep_learning.ncf import NCFModel
from src.data.data_loader import MovieLens100KLoader

# Carrega dados
loader = MovieLens100KLoader('data/raw')
train_data = loader.load()

# Two-Tower Model
two_tower = TwoTowerModel(
    user_embedding_dim=64,
    item_embedding_dim=64,
    user_tower_units=[128, 64, 32],
    epochs=50
)
two_tower.fit(train_data)

# Recomendações
recommendations = two_tower.recommend(user_id=1, n_items=10)
print("Top-10 recomendações:", recommendations)

# NCF Model
ncf = NCFModel(
    mf_embedding_dim=32,
    mlp_embedding_dim=32,
    hidden_units=[128, 64, 32, 16],
    epochs=50
)
ncf.fit(train_data)

# Análise de componentes
analysis = ncf.analyze_model_components(user_id=1, item_id=100)
print("Análise NCF:", analysis)
```

### Exemplo Simples

```bash
# Execute o exemplo interativo
python example_deep_learning.py
```

## 📋 Configuração

O arquivo `config/config.yaml` permite configurar todos os aspectos do framework:

```yaml
# Modelos de Deep Learning
models:
  default_params:
    two_tower:
      user_embedding_dim: 64
      item_embedding_dim: 64
      user_tower_units: [128, 64, 32]
      item_tower_units: [128, 64, 32]
      dropout_rate: 0.2
      learning_rate: 0.001
      epochs: 50
      batch_size: 256
    
    ncf:
      mf_embedding_dim: 32
      mlp_embedding_dim: 32  
      hidden_units: [128, 64, 32, 16]
      epochs: 50
```

## 🔧 Arquitetura dos Modelos de Deep Learning

### Two-Tower Architecture

```
User ID → [User Embedding] → [User Tower] → [User Representation]
                                                     ↓
                                              [Interaction Layer] → Rating
                                                     ↑  
Item ID → [Item Embedding] → [Item Tower] → [Item Representation]
```

**Características:**
- Torres separadas para usuários e itens
- Embeddings independentes permitem melhor escalabilidade
- Combinação por dot product e camadas densas
- Ideal para sistemas de grande escala

### Neural Collaborative Filtering (NCF)

```
User ID → [MF User Emb] ──→ [Element-wise Product] ──┐
                                                      ├→ [Fusion] → Rating
Item ID → [MF Item Emb] ──→ [Element-wise Product] ──┘
          
User ID → [MLP User Emb] ──→ [Concatenate] → [MLP Layers] ──┘
Item ID → [MLP Item Emb] ──→ [Concatenate] → [MLP Layers] ──┘
```

**Características:**
- Combina Matrix Factorization e Multi-Layer Perceptrons
- Embeddings separados para MF e MLP
- Captura interações lineares (MF) e não-lineares (MLP)
- Fusion layer aprende a combinar ambos os componentes

## 📊 Métricas de Avaliação

### Acurácia
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

### Ranking
- **Precision@K**: Precisão nos top-K itens
- **Recall@K**: Recall nos top-K itens  
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision

### Beyond Accuracy
- **Coverage**: Cobertura do catálogo
- **Diversity**: Diversidade intra-lista
- **Novelty**: Novidade das recomendações
- **Personalization**: Personalização

## 📁 Estrutura do Projeto

```
social-recommendation-algorithms/
├── src/
│   ├── data/           # Carregamento e pré-processamento
│   ├── models/         # Implementação dos modelos
│   │   ├── baseline/
│   │   ├── collaborative/ 
│   │   ├── content/
│   │   └── deep_learning/    # 🧠 Modelos de Deep Learning
│   │       ├── two_tower.py
│   │       └── ncf.py
│   ├── evaluation/     # Sistema de avaliação
│   └── utils/          # Utilitários
├── experiments/        # Scripts de experimentos
├── config/            # Configurações
├── data/              # Datasets
└── example_deep_learning.py  # Exemplo interativo
```

## 🎯 Características dos Modelos de Deep Learning

### Vantagens do Two-Tower
- ✅ **Escalabilidade**: Torres independentes permitem paralelização
- ✅ **Interpretabilidade**: Representações separadas de usuários/itens
- ✅ **Flexibilidade**: Diferentes arquiteturas para cada torre
- ✅ **Cold Start**: Melhor para novos usuários/itens

### Vantagens do NCF
- ✅ **Expressividade**: Combina linear (MF) e não-linear (MLP)
- ✅ **Performance**: Estado da arte em datasets clássicos
- ✅ **Análise**: Permite análise das contribuições MF vs MLP
- ✅ **Robustez**: Múltiplos caminhos de aprendizado

## 💻 Requisitos

- Python 3.11+
- TensorFlow 2.14+
- TensorFlow Recommenders 0.7+
- pandas, numpy, scikit-learn
- PyYAML, tqdm, streamlit

## 🚀 Próximos Passos

- [ ] Transformer-based models
- [ ] Autoencoders para recomendação
- [ ] Graph Neural Networks
- [ ] Multi-task learning
- [ ] Reinforcement Learning para recomendação
- [ ] Modelos híbridos avançados

## 📖 Referências

- **Two-Tower**: [Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/pub48840/)
- **NCF**: [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- **Deep Learning for RecSys**: [Deep Learning based Recommender System: A Survey](https://arxiv.org/abs/1707.07435)

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Implemente seguindo os padrões existentes
4. Adicione testes e documentação
5. Envie um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja `LICENSE` para mais detalhes.