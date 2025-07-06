# your code here
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os 

# --- 1. Cargar el Modelo  ---

basedir = os.path.abspath(os.path.dirname(__file__))
wine_model_path = os.path.join(basedir, "..", "models", "k_nearest_neighbor_default_42.joblib")
wine_scaler_path = os.path.join(basedir, "..", "models", "k_nearest_neighbor_default_42.joblib")

try:
    wine_model = joblib.load(wine_model_path)
    wine_scaler = joblib.load(wine_scaler_path)

except Exception as e:
    st.error(f"Error al cargar el modelo o el scaler: {e}")
    st.info(f"Buscando modelo en: {wine_model_path}")
    st.info(f"Buscando scaler en: {wine_scaler_path}")
    st.stop() 
quality_class_dict = {
    0: "Baja Calidad (No tan bueno) 👎",
    1: "Buena Calidad (Recomendado) 👍"
}


st.set_page_config(
    page_title="Clasificador de Calidad de Vino Tinto",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🍷 Clasificador de Calidad de Vino Tinto")
st.markdown("Ingresa las propiedades químicas del vino para predecir su calidad (Baja o Buena).")


st.sidebar.header("Parámetros del Vino")

features_order = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

input_values = {}
for feature in features_order:

    if feature == 'fixed acidity':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=7.4, min_value=4.0, max_value=16.0, step=0.1)
    elif feature == 'volatile acidity':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=0.7, min_value=0.1, max_value=1.5, step=0.01)
    elif feature == 'citric acid':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
    elif feature == 'residual sugar':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=1.9, min_value=0.5, max_value=20.0, step=0.1)
    elif feature == 'chlorides':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=0.076, min_value=0.01, max_value=0.6, step=0.001)
    elif feature == 'free sulfur dioxide':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (mg/l)", value=11.0, min_value=1.0, max_value=72.0, step=1.0)
    elif feature == 'total sulfur dioxide':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (mg/l)", value=34.0, min_value=6.0, max_value=300.0, step=1.0)
    elif feature == 'density':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/cm³)", value=0.9978, min_value=0.99, max_value=1.01, step=0.0001, format="%.4f")
    elif feature == 'pH':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}", value=3.51, min_value=2.7, max_value=4.1, step=0.01)
    elif feature == 'sulphates':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (g/l)", value=0.56, min_value=0.3, max_value=2.0, step=0.01)
    elif feature == 'alcohol':
        input_values[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').title()} (%)", value=9.4, min_value=8.0, max_value=15.0, step=0.1)


input_data = np.array([[input_values[f] for f in features_order]])


if st.button("Predecir Calidad del Vino"):

    scaled_input_data = wine_scaler.transform(input_data)


    prediction_numeric = wine_model.predict(scaled_input_data)[0]

  
    pred_quality = quality_class_dict.get(prediction_numeric, "Desconocido")

  
    st.subheader("Resultado de la Predicción:")
    if prediction_numeric == 1:
        st.success(f"La calidad del vino es: **{pred_quality}**")
    elif prediction_numeric == 0:
        st.warning(f"La calidad del vino es: **{pred_quality}**")
    else:
        st.info(f"La calidad del vino es: **{pred_quality}**")

    st.write("---")
    st.write("Valores Ingresados:")
    st.json(input_values) 