### Importing 
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

### Config 
sns.set_theme(
    context='notebook', style = 'darkgrid',
    font_scale=.75, palette = 'icefire',
    rc= {
        'figure.figsize':(8,4),
        "axes.grid":True, "grid.alpha":.3,
        'axes.titlesize':'large',
        'axes.titleweight':'bold',
        'axes.titlepad':30,'axes.labelpad':15
    }
)
scatter_kwargs = dict(palette='bright', alpha=0.8, linewidth=0)

# Configurating streamlit 
st.set_page_config(layout='wide')

PATH = r'C:\Users\U6094291\Desktop\StockPrice\customer_dataset.csv'

@st.cache_data

def load_data(path:str):
    df = pd.read_csv(path)
    return df

df = load_data(PATH)

st.title("Aircraft Accident Report") 
st.markdown(
    """
    This report has the goal to clearify accidents and incidents which happen with aircrafts under the suppervison of **Brazillian Avation Agency**.   
    """
)
if st.sidebar.checkbox("Show table?"):
    st.header("Raw data")
    st.write(df)

st.sidebar.info("{} lines has been loaded".format(df.shape[0]))

st.subheader("Somenthing")


x = st.sidebar.selectbox("Month", df['preferred_category'].unique())
df_filtred = df[df['preferred_category'] == x]
col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

# fig_date = px.bar(df)