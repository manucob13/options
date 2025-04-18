import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

# Configuración inicial para ampliar el ancho de la página
st.set_page_config(layout="wide")

########################### BACKTESTING 
# --- Cálculos adicionales y Backtesting ---

# [Todo el código anterior de procesamiento de datos se mantiene igual hasta la creación de resumen y tabla_prediccion...]

# Mostrar la tabla de resumen centrada y sin scroll
st.subheader("Resumen de Backtesting")

# Crear estilo para centrado y ancho completo
styles = [
    dict(selector="th", props=[("text-align", "center"),
                               ("font-size", "14px"),
                               ("background-color", "#f0f2f6")]),
    dict(selector="td", props=[("text-align", "center"),
                               ("font-size", "14px")]),
    dict(selector="table", props=[("margin-left", "auto"),
                                  ("margin-right", "auto"),
                                  ("width", "100%")]),
    dict(selector=".col_heading", props=[("text-align", "center")]),
    dict(selector=".row_heading", props=[("display", "none")])
]

# Aplicar estilos y mostrar tabla
styled_resumen = (resumen.style
                  .set_table_styles(styles)
                  .hide_index()
                  .set_properties(**{'width': 'auto', 'max-width': 'auto'}))
st.write(styled_resumen.to_html(), unsafe_allow_html=True)

# --- Predicción del Próximo Día de Negociación ---

# [Código anterior para crear tabla_prediccion...]

st.subheader("Predicción del Próximo Día de Negociación")

# Aplicar mismos estilos a la tabla de predicción
styled_prediccion = (tabla_prediccion.style
                     .set_table_styles(styles)
                     .hide_index()
                     .set_properties(**{'width': 'auto', 'max-width': 'auto'}))
st.write(styled_prediccion.to_html(), unsafe_allow_html=True)

# [El resto del código se mantiene igual...]
