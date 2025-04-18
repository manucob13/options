import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

# --- Configuración de fechas ---
start_date = "2010-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# --- Descargar datos de SPY ---
tickers = ["^GSPC"]
df_spy = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)
df_spy['WMA_30'] = ta.trend.WMAIndicator(close=df_spy['Close'], window=30).wma()
df_spy['log_return'] = np.log(df_spy['Close'] / df_spy['Close'].shift(1))
df_spy['vol_21'] = df_spy['log_return'].rolling(window=21).std()
df_spy['mean_vol_21_252'] = df_spy['vol_21'].rolling(window=252).mean()

# --- Crear dataframe base ---
df2 = pd.DataFrame({
    'Open': df_spy['Open'],
    'Close': df_spy['Close'],
    'Vol21': df_spy['vol_21'],
    'Avg_252_Vol21': df_spy['mean_vol_21_252'],
    'SP500_WMA_30': df_spy['WMA_30']
})

# --- Descargar datos del VIX ---
df_vix = yf.download(["^VIX"], start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)
df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX'})
df_vix['VIX_WMA_21'] = ta.trend.WMAIndicator(close=df_vix['VIX'], window=21).wma()

# --- Unir dataframes y procesar ---
df = df2.join(df_vix, how='left').dropna()
df['Close_y'] = df['Close'].shift(1)
df['Avg_252_Vol21_y'] = df['Avg_252_Vol21'].shift(1)
df['SP500_WMA_30_y'] = df['SP500_WMA_30'].shift(1)
df['VIX_C_y'] = df['VIX'].shift(1)
df['VIX_WMA_21_y'] = df['VIX_WMA_21'].shift(1)
df['VIX_WMA_21_2dy'] = df['VIX_WMA_21'].shift(2)
df = df.dropna()

# --- Calcular bandas 2STD ---
df['2std_DW'] = df['Open'] * (1 - 2 * df['Avg_252_Vol21_y'])
df['2std_UP'] = df['Open'] * (1 + 2 * df['Avg_252_Vol21_y'])

# --- Condiciones ---
df['TREND'] = np.where(df['Close_y'] > df['SP500_WMA_30_y'], 'Alcista', 'Bajista')
df['2_VIX25'] = df['VIX_C_y'] <= 25
df['3_VIX_WMA'] = (
    (df['VIX_C_y'] < df['VIX_WMA_21_y']) &
    (df['VIX_WMA_21_y'] < df['VIX_WMA_21_2dy'])
)
df['Within_2std_252d'] = (df['Close'] >= df['2std_DW']) & (df['Close'] <= df['2std_UP'])

# --- Filtrar y agrupar ---
df_final = df[df['2_VIX25'] & df['3_VIX_WMA']].copy()

# --- Resumen de Backtesting ---
resumen = df_final.groupby('TREND')['Within_2std_252d'].agg(
    Total_Días='count',
    Aciertos='sum'
).reset_index()
resumen['Fallos'] = resumen['Total_Días'] - resumen['Aciertos']
resumen['Winrate(%)'] = round((resumen['Aciertos'] / resumen['Total_Días']) * 100, 2)
resumen = resumen.round({'Total_Días': 2, 'Aciertos': 2, 'Winrate(%)': 2})
resumen = resumen.reset_index(drop=True)

# --- Mostrar tabla resumen con estilo HTML ---
st.subheader("Resumen de Backtesting")
resumen_html = resumen.to_html(index=False, classes='styled-table')
st.markdown(
    """
    <style>
    .styled-table {
        border-collapse: collapse;
        margin: 0 auto;
        font-size: 14px;
        width: auto;
    }
    .styled-table th, .styled-table td {
        border: 1px solid #ddd;
        padding: 6px 10px;
        text-align: left;
        white-space: nowrap;
    }
    .styled-table th {
        background-color: #000000;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown(resumen_html, unsafe_allow_html=True)

# --- Predicción del siguiente día ---
last_row = df.iloc[-1]
last_date = df.index[-1]
next_business_day = last_date + timedelta(days=1)
while next_business_day.weekday() >= 5:
    next_business_day += timedelta(days=1)

# --- Preparar tabla con HTML coloreado ---
tendencia_color = 'green' if last_row['Close_y'] > last_row['SP500_WMA_30_y'] else 'red'
vix25_color = 'green' if last_row['VIX_C_y'] <= 25 else 'red'
vix_wma_color = 'green' if (last_row['VIX_C_y'] < last_row['VIX_WMA_21_y']) and (last_row['VIX_WMA_21_y'] < last_row['VIX_WMA_21_2dy']) else 'red'

tabla_html = f"""
<table class='styled-table'>
    <thead>
        <tr>
            <th>New Date</th>
            <th>Last Close</th>
            <th>Last SP500_WMA_30</th>
            <th>Tendencia</th>
            <th>Last VIX</th>
            <th>Last VIX_WMA_21</th>
            <th>VIX &lt; 25</th>
            <th>VIX &lt; Last 1-2 WMA21</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>{next_business_day.strftime('%Y-%m-%d')}</td>
            <td>{round(last_row['Close_y'], 2)}</td>
            <td>{round(last_row['SP500_WMA_30_y'], 2)}</td>
            <td style="color:{tendencia_color}; font-weight: bold">{'Alcista' if tendencia_color == 'green' else 'Bajista'}</td>
            <td>{round(last_row['VIX_C_y'], 2)}</td>
            <td>{round(last_row['VIX_WMA_21_y'], 2)}</td>
            <td style="color:{vix25_color}; font-weight: bold">{'True' if vix25_color == 'green' else 'False'}</td>
            <td style="color:{vix_wma_color}; font-weight: bold">{'True' if vix_wma_color == 'green' else 'False'}</td>
        </tr>
    </tbody>
</table>
"""

# --- Mostrar tabla de predicción ---
st.subheader("Predicción del Próximo Día de Negociación")
st.markdown(tabla_html, unsafe_allow_html=True)

# --- Cono de probabilidad ---
st.subheader("Cono de Probabilidad")
last_avg_vol_21 = df['Avg_252_Vol21_y'].iloc[-1]
open_value = st.number_input("Valor", min_value=0.0, value=100.0, step=0.01)
std_down = round(open_value * (1 - 2 * last_avg_vol_21), 2)
std_up = round(open_value * (1 + 2 * last_avg_vol_21), 2)
st.markdown(f"**2STD_DOWN**&nbsp;&nbsp;&nbsp;&nbsp;{std_down}")
st.markdown(f"**2STD_UP**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{std_up}")
