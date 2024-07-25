import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

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
colors = px.colors.sequential.Plasma

# Configurating streamlit 
st.set_page_config(layout='wide')

PATH = r'/home/usuario/Forex/customer_data_2.csv'

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

st.subheader("Dataset Analyse")

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
