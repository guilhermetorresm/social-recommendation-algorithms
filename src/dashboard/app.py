# src/dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import json
from datetime import datetime
import sys

# Adiciona o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.result_manager import ResultManager
from src.utils.config import Config


class RecommenderDashboard:
    """Dashboard para análise de sistemas de recomendação."""
    
    def __init__(self):
        self.config = Config()
        self.result_manager = ResultManager(self.config.get('data.results_path'))
        self.setup_page()
    
    def setup_page(self):
        """Configura a página do Streamlit."""
        st.set_page_config(
            page_title="Sistema de Recomendação - Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS customizado
        st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Executa o dashboard."""
        st.title("🎯 Dashboard de Análise - Sistemas de Recomendação")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configurações")
            page = st.selectbox(
                "Selecione a página",
                ["📊 Visão Geral", "🔬 Comparação de Modelos", 
                 "📈 Análise Detalhada", "🧪 Experimentos"]
            )
        
        # Roteamento de páginas
        if page == "📊 Visão Geral":
            self.show_overview()
        elif page == "🔬 Comparação de Modelos":
            self.show_model_comparison()
        elif page == "📈 Análise Detalhada":
            self.show_detailed_analysis()
        elif page == "🧪 Experimentos":
            self.show_experiments()
    
    def show_overview(self):
        """Mostra visão geral dos experimentos."""
        st.header("📊 Visão Geral dos Experimentos")
        
        # Carrega todos os experimentos
        all_experiments = self.result_manager.get_all_experiments()
        
        if all_experiments.empty:
            st.warning("Nenhum experimento encontrado. Execute alguns experimentos primeiro!")
            return
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total de Experimentos",
                len(all_experiments),
                delta=None
            )
        
        with col2:
            st.metric(
                "Modelos Únicos",
                all_experiments['model_name'].nunique(),
                delta=None
            )
        
        with col3:
            if 'rmse' in all_experiments.columns:
                best_rmse = all_experiments['rmse'].min()
                st.metric(
                    "Melhor RMSE",
                    f"{best_rmse:.3f}",
                    delta=None
                )
        
        with col4:
            if 'mae' in all_experiments.columns:
                best_mae = all_experiments['mae'].min()
                st.metric(
                    "Melhor MAE",
                    f"{best_mae:.3f}",
                    delta=None
                )
        
        # Tabela de experimentos recentes
        st.subheader("📋 Experimentos Recentes")
        
        # Prepara dados para exibição
        display_columns = ['experiment_id', 'model_name', 'dataset_name', 'datetime']
        metric_columns = [col for col in ['rmse', 'mae', 'coverage', 'novelty'] 
                         if col in all_experiments.columns]
        
        recent_experiments = all_experiments[display_columns + metric_columns].tail(10)
        recent_experiments = recent_experiments.sort_values('datetime', ascending=False)
        
        # Formata valores numéricos
        for col in metric_columns:
            if col in ['coverage']:
                recent_experiments[col] = recent_experiments[col].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
                )
            else:
                recent_experiments[col] = recent_experiments[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "-"
                )
        
        st.dataframe(
            recent_experiments,
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de evolução
        if len(all_experiments) > 1:
            st.subheader("📈 Evolução das Métricas")
            
            # Seletor de métrica
            available_metrics = [col for col in ['rmse', 'mae', 'coverage', 'novelty'] 
                               if col in all_experiments.columns]
            
            if available_metrics:
                selected_metric = st.selectbox(
                    "Selecione a métrica",
                    available_metrics,
                    format_func=lambda x: x.upper()
                )
                
                # Prepara dados para o gráfico
                plot_data = all_experiments[['datetime', 'model_name', selected_metric]].copy()
                plot_data['datetime'] = pd.to_datetime(plot_data['datetime'])
                plot_data = plot_data.dropna(subset=[selected_metric])
                
                # Cria gráfico
                fig = px.line(
                    plot_data,
                    x='datetime',
                    y=selected_metric,
                    color='model_name',
                    title=f'Evolução da métrica {selected_metric.upper()}',
                    markers=True
                )
                
                fig.update_layout(
                    xaxis_title="Data/Hora",
                    yaxis_title=selected_metric.upper(),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def show_model_comparison(self):
        """Mostra comparação entre modelos."""
        st.header("🔬 Comparação de Modelos")
        
        all_experiments = self.result_manager.get_all_experiments()
        
        if all_experiments.empty:
            st.warning("Nenhum experimento encontrado.")
            return
        
        # Seletor de modelos
        available_models = all_experiments['model_name'].unique()
        selected_models = st.multiselect(
            "Selecione os modelos para comparar",
            available_models,
            default=list(available_models)
        )
        
        if not selected_models:
            st.info("Selecione pelo menos um modelo para visualizar a comparação.")
            return
        
        # Filtra dados
        comparison_data = all_experiments[
            all_experiments['model_name'].isin(selected_models)
        ]
        
        # Agrupa por modelo e calcula médias
        model_stats = comparison_data.groupby('model_name').agg({
            col: 'mean' for col in comparison_data.columns 
            if col in ['rmse', 'mae', 'coverage', 'novelty', 'training_time']
        }).reset_index()
        
        # Gráfico de barras comparativo
        st.subheader("📊 Comparação de Métricas de Erro")
        
        if 'rmse' in model_stats.columns and 'mae' in model_stats.columns:
            fig_errors = go.Figure()
            
            fig_errors.add_trace(go.Bar(
                name='RMSE',
                x=model_stats['model_name'],
                y=model_stats['rmse'],
                text=model_stats['rmse'].round(3),
                textposition='auto',
            ))
            
            fig_errors.add_trace(go.Bar(
                name='MAE',
                x=model_stats['model_name'],
                y=model_stats['mae'],
                text=model_stats['mae'].round(3),
                textposition='auto',
            ))
            
            fig_errors.update_layout(
                title="Métricas de Erro por Modelo",
                xaxis_title="Modelo",
                yaxis_title="Erro",
                barmode='group',
                showlegend=True
            )
            
            st.plotly_chart(fig_errors, use_container_width=True)
        
        # Gráfico radar
        st.subheader("🎯 Análise Multidimensional")
        
        # Prepara dados para o gráfico radar
        radar_metrics = ['rmse', 'mae', 'coverage', 'novelty']
        available_radar_metrics = [m for m in radar_metrics if m in model_stats.columns]
        
        if len(available_radar_metrics) >= 3:
            # Normaliza métricas para escala 0-1
            normalized_data = model_stats.copy()
            for metric in available_radar_metrics:
                if metric in ['rmse', 'mae']:  # Menor é melhor
                    normalized_data[metric] = 1 - (
                        (normalized_data[metric] - normalized_data[metric].min()) / 
                        (normalized_data[metric].max() - normalized_data[metric].min())
                    )
                else:  # Maior é melhor
                    normalized_data[metric] = (
                        (normalized_data[metric] - normalized_data[metric].min()) / 
                        (normalized_data[metric].max() - normalized_data[metric].min())
                    )
            
            fig_radar = go.Figure()
            
            for _, row in normalized_data.iterrows():
                values = [row[metric] for metric in available_radar_metrics]
                values.append(values[0])  # Fecha o polígono
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=available_radar_metrics + [available_radar_metrics[0]],
                    fill='toself',
                    name=row['model_name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Comparação Multidimensional (Valores Normalizados)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Tabela comparativa detalhada
        st.subheader("📋 Tabela Comparativa Detalhada")
        
        # Formata a tabela
        display_stats = model_stats.copy()
        for col in display_stats.columns:
            if col in ['rmse', 'mae', 'novelty']:
                display_stats[col] = display_stats[col].apply(lambda x: f"{x:.3f}")
            elif col == 'coverage':
                display_stats[col] = display_stats[col].apply(lambda x: f"{x*100:.1f}%")
            elif col == 'training_time':
                display_stats[col] = display_stats[col].apply(lambda x: f"{x:.2f}s")
        
        st.dataframe(
            display_stats,
            use_container_width=True,
            hide_index=True
        )
    
    def show_detailed_analysis(self):
        """Mostra análise detalhada de um experimento específico."""
        st.header("📈 Análise Detalhada de Experimento")
        
        all_experiments = self.result_manager.get_all_experiments()
        
        if all_experiments.empty:
            st.warning("Nenhum experimento encontrado.")
            return
        
        # Seletor de experimento
        experiment_options = all_experiments.apply(
            lambda row: f"{row['experiment_id']} - {row['model_name']} ({row['datetime']})",
            axis=1
        ).tolist()
        
        selected_idx = st.selectbox(
            "Selecione um experimento",
            range(len(experiment_options)),
            format_func=lambda x: experiment_options[x]
        )
        
        selected_experiment = all_experiments.iloc[selected_idx]
        
        # Carrega dados completos do experimento
        try:
            experiment_data = self.result_manager.load_experiment_results(
                selected_experiment['results_path']
            )
        except Exception as e:
            st.error(f"Erro ao carregar dados do experimento: {str(e)}")
            return
        
        # Informações do experimento
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Informações Gerais")
            st.write(f"**ID:** {selected_experiment['experiment_id']}")
            st.write(f"**Modelo:** {selected_experiment['model_name']}")
            st.write(f"**Dataset:** {selected_experiment['dataset_name']}")
            st.write(f"**Data/Hora:** {selected_experiment['datetime']}")
        
        with col2:
            st.subheader("📊 Métricas Principais")
            
            metrics_cols = st.columns(4)
            metric_names = ['rmse', 'mae', 'coverage', 'novelty']
            
            for i, metric in enumerate(metric_names):
                if metric in experiment_data['metrics']:
                    value = experiment_data['metrics'][metric]
                    with metrics_cols[i]:
                        if metric == 'coverage':
                            st.metric(metric.upper(), f"{value*100:.1f}%")
                        else:
                            st.metric(metric.upper(), f"{value:.3f}")
        
        # Métricas de ranking se disponíveis
        if 'ranking' in experiment_data['metrics']:
            st.subheader("🎯 Métricas de Ranking")
            
            ranking_data = []
            for k_value, metrics in experiment_data['metrics']['ranking'].items():
                row = {'k': k_value.replace('at_', '')}
                row.update(metrics)
                ranking_data.append(row)
            
            ranking_df = pd.DataFrame(ranking_data)
            
            # Gráfico de métricas de ranking
            fig_ranking = go.Figure()
            
            for metric in ['precision', 'recall', 'map']:
                if metric in ranking_df.columns:
                    fig_ranking.add_trace(go.Scatter(
                        x=ranking_df['k'],
                        y=ranking_df[metric],
                        mode='lines+markers',
                        name=metric.capitalize(),
                        text=ranking_df[metric].round(3),
                        textposition="top center"
                    ))
            
            fig_ranking.update_layout(
                title="Métricas de Ranking por k",
                xaxis_title="k",
                yaxis_title="Valor",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ranking, use_container_width=True)
        
        # Parâmetros do modelo
        if 'model_params' in experiment_data['metadata']:
            st.subheader("⚙️ Parâmetros do Modelo")
            st.json(experiment_data['metadata']['model_params'])
    
    def show_experiments(self):
        """Mostra interface para novos experimentos."""
        st.header("🧪 Executar Novo Experimento")
        
        st.info("Esta funcionalidade permite executar novos experimentos diretamente do dashboard.")
        
        # Formulário de configuração
        with st.form("experiment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                dataset = st.selectbox(
                    "Dataset",
                    ["movielens-100k", "movielens-1m", "lastfm", "book-crossing"]
                )
                
                models = st.multiselect(
                    "Modelos",
                    ["global_mean", "popularity_count", "popularity_rating", 
                     "knn_user", "knn_item", "svd", "als"],
                    default=["global_mean", "popularity_count"]
                )
            
            with col2:
                test_size = st.slider(
                    "Tamanho do conjunto de teste (%)",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5
                ) / 100
                
                random_seed = st.number_input(
                    "Seed aleatória",
                    value=42,
                    step=1
                )
            
            submit_button = st.form_submit_button("🚀 Executar Experimento")
        
        if submit_button:
            st.warning("⚠️ Funcionalidade em desenvolvimento!")
            st.info("""
            Para executar experimentos, use o script de linha de comando:
            ```bash
            python experiments/run_experiments.py --mode baselines --dataset movielens-100k
            ```
            """)
        
        # Instruções
        st.subheader("📖 Como executar experimentos")
        
        st.markdown("""
        ### Via linha de comando:
        
        1. **Executar modelos baseline:**
           ```bash
           python experiments/run_experiments.py --mode baselines
           ```
        
        2. **Executar modelos específicos:**
           ```bash
           python experiments/run_experiments.py --models global_mean popularity_count
           ```
        
        3. **Usar dataset diferente:**
           ```bash
           python experiments/run_experiments.py --dataset movielens-1m
           ```
        
        ### Resultados:
        
        Os resultados serão salvos automaticamente em `data/results/` e aparecerão neste dashboard.
        """)


def main():
    """Função principal do dashboard."""
    dashboard = RecommenderDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()