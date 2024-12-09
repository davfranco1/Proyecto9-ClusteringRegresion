import streamlit as st
import pandas as pd
import pickle
import time

from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Predicción de Beneficio para el Cluster 0",
    page_icon="",
    layout="centered",
)

# Título y descripción
st.title("Predicción de Beneficios generados por clientes por el cluster 0")
st.write("Esta aplicación permite predecir el beneficio de una venta basándonos en sus características.")

# Mostrar una imagen llamativa
st.image(
    "../images/header.jpg",  # URL de la imagen
    #caption="",
    use_container_width=True,
)

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('../transformers/cluster0_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    with open('../transformers/cluster0_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('../transformers/cluster0_model.pkl', 'rb') as f:
        model = pickle.load(f)

    return encoder, scaler, model

encoder, scaler, model = load_models()

# Formularios de entrada
st.header("🔧 Características de la operación")
col1, col2 = st.columns(2)

with col1:
    shipmode = st.text_input("Modo de Envío")
    country = st.text_input("País")
    market = st.text_input("Mercado")
    region = st.text_input("Región")

with col2:
    category = st.text_input("Categoría")
    subcat = st.text_input("Subcategoría")
    orderpriority = st.text_input("Prioridad de Pedido")
    quantity = st.number_input("Cantidad", min_value=1, step=1)
    discount = st.number_input("Descuento", min_value=0.0, max_value=1.0, step=0.01)

# Botón para realizar la predicción
if st.button("💡 Predecir Precio"):
    # Crear DataFrame con los datos ingresados
    nuevo_cliente = pd.DataFrame({
        'Ship Mode': [shipmode], 
        'Country': [country],
        'Market': [market],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [subcat],
        'Order Priority': [orderpriority],
        'Quantity': [quantity],
        'Discount': [discount]
    })

    df_new = pd.DataFrame(nuevo_cliente)
    df_new = df_new[['Ship Mode', 'Country', 'Market', 'Region', 'Category',
        'Sub-Category', 'Quantity', 'Discount', 'Order Priority']]
    df_new

    df_pred = df_new.copy()

    df_pred = encoder.transform(df_pred)

    col_num = ['Ship Mode', 'Country', 'Market', 'Region', 'Category',
       'Sub-Category', 'Quantity', 'Discount', 'Order Priority']
    df_pred[col_num] = scaler.transform(df_pred[col_num])

    # Realizar la predicción
    prediction = round(model.predict(df_pred)[0], 0)

    # Mostrar el resultado
    with st.spinner('Estamos calculando el beneficio de la operación...'):
        time.sleep(3)
    st.success(f"💵 El beneficio estimado de la operación es {prediction}€")
