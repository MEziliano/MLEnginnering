import os, sys, platform
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

### Config 
sns.set_theme(
    context='notebook', style='darkgrid',
    font_scale=.75, 
    rc={
        'figure.figsize': (14, 7),
        "axes.grid": True, "grid.alpha": .3,
        'axes.titlesize': 'large',
        'axes.titleweight': 'bold',
        'axes.titlepad': 30, 'axes.labelpad': 15
    }
) # palette = 'icefire',
scatter_kwargs = dict(palette='bright', alpha=0.8, linewidth=0)

color_palettes = {
    'Plasma': px.colors.sequential.Plasma,
    'Viridis': px.colors.sequential.Viridis,
    'Cividis': px.colors.sequential.Cividis,
    'Colorblind': sns.color_palette("colorblind"),
}

# Configurating streamlit 
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)


PATH = r"C:\Users\U6094291\Desktop\StockPrice\customer_data_2.csv"if platform.system() == "Windows" else r"/home/usuario/Forex/customer_data_2.csv"

@st.cache_data
def load_data(path: str):
    # Function to load data
    df = pd.read_csv(path)

    mapa_genero = {
        'M': 'Male',
        'male': 'Male',
        'F': 'Female',
        'female': 'Female',
        'Male': 'Male',
        'Female': 'Female',
        'Other': 'Other'
    }
    mapa_categoria = {
        'Elec': 'Electronics',
        'Sports': "Sports",
        'Electronics': "Electronics",
        'Home & Garden': "Home & Garden",
        'Groceries': "Groceries",
        'Clothing': "Clothing"
    }

    df['gender'] = df['gender'].map(mapa_genero)
    df['preferred_category'] = df['preferred_category'].map(mapa_categoria)
    df.drop(columns='id', axis=1, inplace=True)
    # df.dropna(axis=0, inplace=True)
    return df

df = load_data(PATH)

st.title("Fi Group Costumer Analysis Report") 
st.markdown(
    """
    This report has the goal to analyse the costumers of **FI Group**.   
    """)

with st.sidebar:
    st.title("Dataset info")
    if st.checkbox("Check the raw data"):
        st.write(df)

    st.info(f"{df.shape[0]} lines has been loaded")
    st.subheader("**Null values in the DataFrame**")
    st.table(pd.DataFrame((df.isnull().sum().sort_values(ascending=False) / len(df)) * 100).style.bar(color='red'))

    st.info(f"{df.duplicated().sum()} lines have been found as duplicated")
    st.subheader("**Unique values for each column**")
    st.table(pd.DataFrame(df.nunique().sort_values(ascending=False)).style.bar(color='darkblue'))

    # Seletor de paleta de cores
    selected_palette = st.sidebar.selectbox("Select color palette", options=list(color_palettes.keys()))
    colors = color_palettes[selected_palette]

tab1, tab2 = st.tabs(["Analysis", "Machine Learning - Cluster"])

with tab1:
    st.header("Dataset Analyse")

    # Ajuste o layout do gráfico para uma visualização mais agradável
    def adjust_fig_layout(fig):
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),  # Ajuste as margens
            height=400,  # Altura do gráfico
            plot_bgcolor='rgba(0,0,0,0)',  # Cor de fundo transparente
            paper_bgcolor='rgba(0,0,0,0)',  # Cor de fundo do papel transparente
        )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    data_1 = pd.crosstab(df['gender'], df['preferred_category']).reset_index()
    chart_1 = px.bar(data_1, x=data_1.columns[1:], y='gender', title="Consumption across gender and categories", orientation='h', barmode='stack', text_auto=True, color_discrete_sequence=colors)
    adjust_fig_layout(chart_1)
    col1.plotly_chart(chart_1, use_container_width=True)

    data_2 = pd.crosstab(df['preferred_category'], df['gender']).reset_index()
    chart_2 = px.bar(data_2, x='preferred_category', y=data_2.columns, title="Consumption across categories and gender", barmode='stack', text_auto=True, color_discrete_sequence=colors)
    adjust_fig_layout(chart_2)
    col2.plotly_chart(chart_2, use_container_width=True)

    col3.subheader("**Income according with the gender**")
    col3.table(pd.DataFrame(data=df.groupby(df['gender']).agg({"income": ['min', 'mean', 'median', 'max', 'sum']})).style.background_gradient())

    col4.subheader("**Spending category according with the category**")
    col4.table(pd.DataFrame(data=df.groupby(df['preferred_category']).agg({"spending_score": ['min', 'mean', 'median', 'max', 'sum']})).style.background_gradient())

with tab2:
    st.header("Cluster Analysis")

    X = (df
     .dropna()  # Remove linhas com valores ausentes
     .assign(
         gender=lambda x: x['gender'].map({'Female': 0, 'Male': 1, 'Other': 2}),
         preferred_category=lambda x: pd.factorize(x['preferred_category'])[0]))

    features = X[['age', 'gender', 'preferred_category', 'income']]  # Ajuste conforme necessário

    # Normalizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    option = st.radio(
        "Which technique  do you would like to use?",
        ["KMeans", "DBScan"])

    st.write("You selected:", option)

    if option == 'KMeans': 
        col1, col2 = st.columns(2)
        with col1:
            clusters = st.slider("Select the number of the clusters",
                    2, 11)
            st.write("Values:", clusters)
            n_clusters = clusters  # Ajuste conforme necessário
            features = st.multiselect("Which features would like to add?", ['age', 'gender', 'preferred_category', 'income'] )

            # Aplicar o K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)

            # Adicionar os clusters ao DataFrame original
            X['cluster'] = clusters

            # Exemplo de visualização se você tiver apenas duas características
            def plot_kmeans_clusters(X):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=X['gender'], y=X['preferred_category'], hue=X['cluster'], palette='viridis', ax=ax)
                ax.set_title('Clusters de K-Means')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Preferred Category')
                ax.legend(title='Cluster')
                return fig

            col1.pyplot(plot_kmeans_clusters(X))

            cluster_metrics = silhouette_score, davies_bouldin_score, calinski_harabasz_score
            cluster_metrics_results = []

            for k in range(2,11):
                model = KMeans(n_clusters=k, random_state=0)
                labels = model.fit_predict(X)
                cluster_results_dict = {'k':k}
                cluster_results_dict['inertia'] = model.inertia_
                for metric in cluster_metrics:
                    cluster_results_dict[metric.__name__] = metric(X, labels)
                cluster_metrics_results.append(cluster_results_dict)

        with col2:

            st.table(pd.DataFrame(cluster_metrics_results).set_index('k').style.background_gradient(cmap='Blues'))

    else:
        # Treinamento do modelo DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(features_scaled)
        X['cluster_SCAN'] = clusters

        # Função para plotar os resultados do DBSCAN
        def plot_dbscan_clusters(X):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X['gender'], y=X['preferred_category'], hue=X['cluster_SCAN'], palette='viridis', ax=ax)
            ax.set_title('Clusters de DBSCAN')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Preferred Category')
            ax.legend(title='Cluster')
            return fig

        # Cálculo das métricas dos clusters
        cluster_metrics_results = []
        for cluster in np.unique(clusters):
            if cluster != -1:  # Ignorar o cluster de ruído
                cluster_data = X[X['cluster_SCAN'] == cluster]
                metrics = {
                    'cluster': cluster,
                    'count': len(cluster_data),
                    'mean_income': cluster_data['income'].mean(),
                    'median_income': cluster_data['income'].median(),
                    'min_income': cluster_data['income'].min(),
                    'max_income': cluster_data['income'].max()
                }
                cluster_metrics_results.append(metrics)

        # Converter as métricas em um DataFrame
        metrics_df = pd.DataFrame(cluster_metrics_results).set_index('cluster')

        # Criar colunas
        col1, col2 = st.columns(2)

        # Exibir o gráfico na primeira coluna
        with col1:
            st.title("Clusters de DBSCAN")
            fig = plot_dbscan_clusters(X)
            st.pyplot(fig)

        # Exibir a tabela na segunda coluna
        with col2:
            st.title("Métricas dos Clusters")
            st.table(metrics_df.style.background_gradient(cmap='Blues'))

