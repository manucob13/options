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
ventana_vol = st.number_input("Ventana de Volatilidad (días - default 21)", min_value=5, max_value=252, value=21)
anio_inicio = st.number_input("Año de Inicio", min_value=2000, max_value=datetime.today().year, value=2020)

# --- Función para obtener datos ---
@st.cache_data
def market_data(ticker, start_date, end_date, interval='1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)

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

# Volatilidad en porcentaje
data['vol'] = data['log_return'].rolling(window=ventana_vol).std() * 100
data = data.dropna()

# --- Datos temporales para resampling ---
data['year'] = data.index.year
data['month'] = data.index.month

# --- Volatilidad mensual y anual ---
monthly_vol = data[data['year'] >= anio_inicio].groupby(['year', 'month'])['vol'].mean().unstack()
annual_vol = data[data['year'] >= anio_inicio].groupby('year')['vol'].mean()
monthly_vol['Anual'] = annual_vol
monthly_vol = monthly_vol[[1,2,3,4,5,6,7,8,9,10,11,12,'Anual']]
monthly_vol.columns = ['En', 'Fe', 'Mr', 'Ab', 'My', 'Jn', 
                       'Jl', 'Ag', 'Sp', 'Oc', 'Nv', 'Dc', 'Anual']

# Reemplazar NaNs por cadenas vacías para que no se muestre None
monthly_vol_display = monthly_vol.copy()
monthly_vol_display = monthly_vol_display.applymap(lambda x: "" if pd.isna(x) else f"{x:.2f}")

st.dataframe(monthly_vol_display, use_container_width=True)

# --- Crear monthly_long ---
monthly_long = monthly_vol.reset_index().melt(id_vars=['year'], value_vars=monthly_vol.columns[:-1])
monthly_long.columns = ['year', 'Mes', 'Volatilidad']
orden_meses = ['En', 'Fe', 'Mr', 'Ab', 'My', 'Jn', 'Jl', 'Ag', 'Sp', 'Oc', 'Nv', 'Dc']
monthly_long['Mes'] = pd.Categorical(monthly_long['Mes'], categories=orden_meses, ordered=True)

# --- Promedio mensual continuo ---
vol_mm = data['vol'].resample('ME').mean()
vol_mm = vol_mm[vol_mm.index >= f"{anio_inicio}-01-01"]

# --- Gráfico 1: Volatilidad Anual ---
fig1, ax1 = plt.subplots(figsize=(15, 8))
monthly_vol['Anual'] = pd.to_numeric(monthly_vol['Anual'], errors='coerce')
monthly_vol['Anual'].dropna().plot(kind='bar', color='steelblue', ax=ax1)
ax1.set_title(f'Volatilidad Anual Promedio ({ventana_vol}d) - {ticker} {anio_inicio}-Today')
ax1.set_ylabel('Volatilidad (%)')
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
ax2.set_ylabel('Volatilidad Promedio (%)')
ax2.legend(title='Año', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
st.pyplot(fig2)

# --- Gráfico 3: Volatilidad mensual promedio continua ---
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.plot(vol_mm.index, vol_mm.values, marker='o', linestyle='-', color='darkblue', linewidth=2, alpha=0.9)
ax3.set_title(f'Volatilidad Mensual Promedio ({ventana_vol}d) - {ticker} {anio_inicio}-Today', fontsize=16)
ax3.set_ylabel('Volatilidad (%)')
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

########################### BACKTESTING & PREDICTION ###############
# --- Cálculos adicionales y Backtesting ---

import ta
from datetime import timedelta

# Definir fechas (último año)
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Obtener datos de SPY
tickers = ["SPY"]
df_spy = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)
df_spy['WMA_30'] = ta.trend.WMAIndicator(close=df_spy['Close'], window=30).wma()
# Obtener la volatilidad
df_spy['log_return'] = np.log(df_spy['Close'] / df_spy['Close'].shift(1))
df_spy['vol_21'] = df_spy['log_return'].rolling(window=21).std()
# Calcular la media de la volatilidad en una ventana deslizante de 252 días
df_spy['mean_vol_21_252'] = df_spy['vol_21'].rolling(window=252).mean()

# Crear un dataframe con los datos de interés
df2 = pd.DataFrame({
    'Open'                :   df_spy['Open'],
    'Close'               :   df_spy['Close'],
    'Vol21'               :   df_spy['vol_21'],
    'Avg_252_Vol21'       :   df_spy['mean_vol_21_252'],
    'SP500_WMA_30'        :   df_spy['WMA_30']
})

# Obtener datos del VIX
tickers_vix = ["^VIX"]
df_vix = yf.download(tickers_vix, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)
df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX'})
df_vix['VIX_WMA_21'] = ta.trend.WMAIndicator(close=df_vix['VIX'], window=21).wma()

# Unimos los df
df   = df2.join(df_vix, how='left')
df   = df.dropna()

# Desplazar la avg volatilidad un dia atras - WMA30
df['Close_y'] = df['Close'].shift(1)
df['Avg_252_Vol21_y'] = df['Avg_252_Vol21'].shift(1)
df['SP500_WMA_30_y'] = df['SP500_WMA_30'].shift(1)
df['VIX_C_y'] = df['VIX'].shift(1)
df['VIX_WMA_21_y'] = df['VIX_WMA_21'].shift(1)  # VIX WMA 21 de ayer
df['VIX_WMA_21_2dy'] = df['VIX_WMA_21'].shift(2)  # VIX WMA 21 de hace 2 días
df   = df.dropna()

# Calculo de las bandas 2std
df['2std_DW'] = df['Open'] * (1 - 2 * df['Avg_252_Vol21_y'])
df['2std_UP'] = df['Open'] * (1 + 2 * df['Avg_252_Vol21_y'])

# Tendencia
df['1_TREND'] = np.where(df['Close_y'] > df['SP500_WMA_30_y'], 'Alcista', 'Bajista')

# Baja volatilidad en el VIX < 25
df['2_VIX25'] = df['VIX_C_y'] <= 25

# VIX por debajo de su media 21 en los dos ultimos dias
df['3_VIX_WMA'] = (
    (df['VIX_C_y'] < df['VIX_WMA_21_y']) &  # VIX de ayer < VIX WMA 21 de ayer
    (df['VIX_WMA_21_y'] < df['VIX_WMA_21_2dy'])  # VIX WMA 21 de ayer < VIX WMA 21 de hace 2 días
)

# --- Backtest y Tabla de Resumen ---

# Anadimos columna para ver si ha entrado en rango
df['Within_2std_252d'] = (df['Close'] >= df['2std_DW']) & (df['Close'] <= df['2std_UP'])

# Filtrar las condiciones
df_final = df[
    df['2_VIX25'] & 
    df['3_VIX_WMA']
].copy()

resumen = df_final.groupby('1_TREND')['Within_2std_252d'].agg(
    Total_Días='count',
    Aciertos='sum'
).reset_index()

resumen['Fallos'] = resumen['Total_Días'] - resumen['Aciertos']
resumen['Winrate(%)'] = round(
    (resumen['Aciertos'] / resumen['Total_Días']) * 100, 2
)

# Formatear la tabla para centrar los valores y redondear a 2 decimales
resumen = resumen.round({'Total_Días': 2, 'Aciertos': 2, 'Winrate(%)': 2})

st.subheader("Resumen de Backtesting")
st.table(resumen.style.set_properties(
    **{'text-align': 'center'}
))

# --- Predicción del Próximo Día de Negociación ---

# Tomamos la última fila del dataframe
last_row = df.iloc[-1]
last_date = df.index[-1]
next_business_day = last_date + timedelta(days=1)

# Aseguramos que sea un día hábil (lunes a viernes)
while next_business_day.weekday() >= 5:
    next_business_day += timedelta(days=1)

# Creamos una fila con los datos y redondeamos a 2 decimales
tabla_prediccion = pd.DataFrame([{
    'New Date': next_business_day.strftime('%Y-%m-%d'),
    'Last Close': round(last_row['Close_y'], 2),
    'Last SP500_WMA_30': round(last_row['SP500_WMA_30_y'], 2),
    'Tendencia': 'Alcista' if last_row['Close_y'] > last_row['SP500_WMA_30_y'] else 'Bajista',
    'Last VIX': round(last_row['VIX_C_y'], 2),
    'Last VIX_WMA_21': round(last_row['VIX_WMA_21_y'], 2),
    'VIX < 25': last_row['VIX_C_y'] <= 25,
    'VIX < Last 1-2 WMA21': (
        last_row['VIX_C_y'] < last_row['VIX_WMA_21_y'] and 
        last_row['VIX_WMA_21_y'] < last_row['VIX_WMA_21_2dy']
    )
}])

# Formatear la tabla de predicción para centrar los valores, redondear a 2 decimales, y hacerla más pequeña
tabla_prediccion = tabla_prediccion.round(2)

# Estilo para hacer la tabla más compacta
tabla_prediccion = tabla_prediccion.style.set_properties(
    **{'text-align': 'center', 'font-size': '12px', 'padding': '5px'}
)

st.subheader("Predicción del Próximo Día de Negociación")
st.table(tabla_prediccion)

### TOOL CALCULAR BANDAS SUPERIOR E INFERIOR

# Código de acceso (puedes cambiarlo a lo que desees)
codigo_secreto = "1972026319"

# Pedir al usuario que ingrese el código
codigo_ingresado = st.text_input("Valor:")

# Verificar si el código es correcto
if codigo_ingresado == codigo_secreto:
    # Tomar el último valor de Avg_252_Vol21_y
    last_avg_vol_21 = df['Avg_252_Vol21_y'].iloc[-1]

    # Crear un widget en Streamlit para ingresar el valor de Open
    open_value = st.number_input("SPY Open value", min_value=0.0, value=100.0, step=0.01)

    # Calcular las bandas 2std con el valor de Open ingresado
    std_down = open_value * (1 - 2 * last_avg_vol_21)
    std_up = open_value * (1 + 2 * last_avg_vol_21)

    # Mostrar los resultados
    st.write(f"Con un valor de Open de {open_value}, las bandas 2std son:")
    st.write(f"2std_DOWN: {round(std_down, 2)}")
    st.write(f"2std_UP: {round(std_up, 2)}")
else:
    # Mostrar mensaje de acceso denegado
    st.write("Seccion en construccion.")
