import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

########################### BACKTESTING 
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
    df['2_VIX25'] & df['3_VIX_WMA']
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

# Eliminar el índice (esto es la clave)
resumen = resumen.reset_index(drop=True)

# Mostrar la tabla de resumen sin el índice usando st.dataframe
st.subheader("Resumen de Backtesting")
st.dataframe(resumen)

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
    'VIX < 25': 'True' if last_row['VIX_C_y'] <= 25 else 'False',  # Convertir a string 'True'/'False'
    'VIX < Last 1-2 WMA21': 'True' if (
        last_row['VIX_C_y'] < last_row['VIX_WMA_21_y'] and 
        last_row['VIX_WMA_21_y'] < last_row['VIX_WMA_21_2dy']
    ) else 'False'  # Convertir a string 'True'/'False'
}])

# Eliminar el índice
tabla_prediccion = tabla_prediccion.reset_index(drop=True)

# Mostrar la tabla de predicción sin índice usando st.dataframe
st.subheader("Predicción del Próximo Día de Negociación")
st.dataframe(tabla_prediccion)

### TOOL CALCULAR BANDAS SUPERIOR E INFERIOR

# =========================
# CÁLCULO INTERACTIVO CON ACCESO
# =========================

# Código secreto (puedes modificarlo)
codigo_secreto = "1972026319"

# Entrada de código oculta
codigo_ingresado = st.text_input("Introduce el código para acceder al cálculo interactivo:", type="password")

if codigo_ingresado == codigo_secreto:
    st.markdown("### Cálculo de bandas 2std")

    # Último valor de la media de volatilidad
    last_avg_vol_21 = df['Avg_252_Vol21_y'].iloc[-1]

    # Ingreso interactivo de Open
    open_value = st.number_input("Introduce el valor de Open", min_value=0.0, value=100.0, step=0.01)

    # Cálculo de bandas
    std_down = round(open_value * (1 - 2 * last_avg_vol_21), 2)
    std_up = round(open_value * (1 + 2 * last_avg_vol_21), 2)

    # Mostrar con espaciado elegante
    st.markdown(f"**2STD_DOWN**&nbsp;&nbsp;&nbsp;&nbsp;{std_down}")
    st.markdown(f"**2STD_UP**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{std_up}")
else:
    st.warning("Sección en construcción")
