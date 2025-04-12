import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de estilo
sns.set_theme(style='whitegrid')
st.set_page_config(layout="centered", page_title="Volatilidad Histórica")

# --- Entradas de usuario ---
st.title("📈 Volatilidad Histórica del Mercado")

ticker = st.text_input("Instrumento (Ticker)", value="SPY")
ventana_vol = st.number_input("Ventana de Volatilidad (días)", min_value=5, max_value=252, value=21)
anio_inicio = st.number_input("Año de Inicio", min_value=2000, max_value=datetime.today().year, value=2020)

# --- Función para obtener datos ---
@st.cache_data
def market_data(ticker, start_date, end_date, interval='1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)

    # Si el resultado tiene múltiples columnas (MultiIndex), toma solo 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df[['Close']]

    df = df.rename(columns={"Close": ticker})
    return df

# --- Obtener datos ---
start_date = f"{anio_inicio}-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')
data = market_data(ticker, start_date, end_date)
data['log_return'] = np.log(data[ticker] / data[ticker].shift(1))
data['vol'] = data['log_return'].rolling(window=ventana_vol).std()
data = data.dropna()

# --- Datos temporales para resampling ---
data['year'] = data.index.year
data['month'] = data.index.month

# --- Volatilidad mensual y anual ---
monthly_vol = data[data['year'] >= anio_inicio].groupby(['year', 'month'])['vol'].mean().unstack()
annual_vol = data[data['year'] >= anio_inicio].groupby('year')['vol'].mean()
monthly_vol['Anual'] = annual_vol
monthly_vol = monthly_vol[[1,2,3,4,5,6,7,8,9,10,11,12,'Anual']]
monthly_vol.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic', 'Anual']

st.dataframe(monthly_vol.style.format("{:.2%}"), use_container_width=True)

# --- Crear monthly_long ---
monthly_long = monthly_vol.reset_index().melt(id_vars=['year'], value_vars=monthly_vol.columns[:-1])
monthly_long.columns = ['year', 'Mes', 'Volatilidad']
orden_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
monthly_long['Mes'] = pd.Categorical(monthly_long['Mes'], categories=orden_meses, ordered=True)

# --- Promedio mensual continuo
vol_mm = data['vol'].resample('ME').mean()
vol_mm = vol_mm[vol_mm.index >= f"{anio_inicio}-01-01"]

# --- Gráfico 1: Volatilidad Anual ---
fig1, ax1 = plt.subplots(figsize=(12, 7))
monthly_vol['Anual'] = pd.to_numeric(monthly_vol['Anual'], errors='coerce')
monthly_vol['Anual'].dropna().plot(kind='bar', color='steelblue', ax=ax1)
ax1.set_title(f'Volatilidad Anual Promedio ({ventana_vol}d) - {ticker} {anio_inicio}-Today')
ax1.set_ylabel('Volatilidad')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
st.pyplot(fig1)

# --- Gráfico 2: Volatilidad mensual por año ---
ultimo_anio = monthly_long['year'].max()
anios_previos = sorted(monthly_long['year'].unique())
paleta_colores = sns.color_palette('Set2', len(anios_previos) - 1)
colores = {}
i = 0
for y in anios_previos:
    colores[y] = 'black' if y == ultimo_anio else paleta_colores[i]
    if y != ultimo_anio: i += 1

fig2, ax2 = plt.subplots(figsize=(6, 3))
for year in anios_previos:
    df_linea = monthly_long[monthly_long['year'] == year]
    ax2.plot(df_linea['Mes'], df_linea['Volatilidad'], marker='o', label=str(year),
             color=colores[year], linewidth=2)
ax2.set_title(f'Volatilidad Mensual ({ventana_vol}d) - {ticker} {anio_inicio}-Today', fontsize=16)
ax2.set_xlabel('Mes')
ax2.set_ylabel('Volatilidad Promedio')
ax2.legend(title='Año', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
st.pyplot(fig2)

# --- Gráfico 3: Volatilidad mensual promedio continua ---
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.plot(vol_mm.index, vol_mm.values, marker='o', linestyle='-', color='darkblue', linewidth=2, alpha=0.9)
ax3.set_title(f'Volatilidad Mensual Promedio ({ventana_vol}d) - {ticker} {anio_inicio}-Today', fontsize=16)
ax3.set_ylabel('Volatilidad')
ax3.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
st.pyplot(fig3)

# --- Gráfico 4: Precio del instrumento ---
fig4, ax4 = plt.subplots(figsize=(6, 3))
data1 = data[data['year'] >= anio_inicio]
ax4.plot(data1.index, data1[ticker], color='black', linewidth=1, alpha=0.8)
ax4.set_title(f'{ticker} {anio_inicio}-Today', fontsize=16)
ax4.set_ylabel('Precio de Cierre')
ax4.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
st.pyplot(fig4)

