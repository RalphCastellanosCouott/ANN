import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Predictor de Crédito", page_icon="💳")

@st.cache_resource
def load_assets():
    """Carga todos los archivos necesarios"""
    try:
        # Verificar archivos
        files = os.listdir('.')
        st.write("Archivos encontrados:", files)
        
        # Cargar modelo
        model = load_model('modelo_credito.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Cargar otros archivos
        scaler = joblib.load('minmax_scaler.joblib')
        pca = joblib.load('pca_model.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        
        return model, scaler, pca, label_encoders
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None

# Cargar todo
model, scaler, pca, label_encoders = load_assets()

if model is None:
    st.stop()

st.title("💳 Clasificador de Crédito")
st.write("Ingrese los datos del cliente:")

# Formulario - TODOS los nombres en ESPAÑOL
with st.form("form"):
    col1, col2 = st.columns(2)
    
    with col1:
        num_cuentas = st.number_input("Número de Cuentas Bancarias", 0, 20, 3)
        num_tarjetas = st.number_input("Número de Tarjetas de Crédito", 0, 20, 2)
        tasa_interes = st.number_input("Tasa de Interés (%)", 0, 50, 12)
        retraso = st.number_input("Días de Retraso desde Vencimiento", 0, 100, 5)
        pagos_retrasados = st.number_input("Número de Pagos Retrasados", 0, 50, 2)
    
    with col2:
        num_consultas = st.number_input("Número de Consultas de Crédito", 0, 30, 2)
        deuda_pendiente = st.number_input("Deuda Pendiente ($)", 0.0, 10000.0, 1500.0)
        antiguedad = st.number_input("Antigüedad del Historial Crediticio (años)", 0.0, 40.0, 5.0)
        mezcla_credito = st.selectbox("Mezcla de Crédito", ["Good", "Standard", "Bad"])
        pago_minimo = st.selectbox("Pago del Monto Mínimo", ["Yes", "No"])
    
    submit = st.form_submit_button("Predecir")

if submit:
    try:
        # Preparar datos - usando los mismos nombres que en el entrenamiento (ESPAÑOL)
        input_data = pd.DataFrame({
            'Num_Cuentas_Bancarias': [float(num_cuentas)],
            'Num_Tarjetas_Credito': [float(num_tarjetas)],
            'Tasa_Interes': [float(tasa_interes)],
            'Retraso_Desde_Vencimiento': [float(retraso)],
            'Num_Pagos_Retrasados': [float(pagos_retrasados)],
            'Num_Consultas_Credito': [float(num_consultas)],
            'Mezcla_Credito': [mezcla_credito],
            'Deuda_Pendiente': [float(deuda_pendiente)],
            'Antiguedad_Historial_Crediticio': [float(antiguedad)],
            'Pago_Monto_Minimo': [pago_minimo]
        })
        
        st.write("📋 Datos ingresados:", input_data)
        
        # Aplicar LabelEncoder a las variables categóricas
        # Mezcla_Credito
        if 'Mezcla_Credito' in label_encoders:
            input_data['Mezcla_Credito'] = label_encoders['Mezcla_Credito'].transform(input_data['Mezcla_Credito'])
        else:
            # Fallback manual
            mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
            input_data['Mezcla_Credito'] = input_data['Mezcla_Credito'].map(mix_map)
        
        # Pago_Monto_Minimo
        if 'Pago_Monto_Minimo' in label_encoders:
            input_data['Pago_Monto_Minimo'] = label_encoders['Pago_Monto_Minimo'].transform(input_data['Pago_Monto_Minimo'])
        else:
            # Fallback manual
            payment_map = {'No': 0, 'Yes': 1}
            input_data['Pago_Monto_Minimo'] = input_data['Pago_Monto_Minimo'].map(payment_map)
        
        st.write("🔢 Datos codificados:", input_data)
        
        # Escalar
        X_scaled = scaler.transform(input_data)
        
        # Aplicar PCA
        X_pca = pca.transform(X_scaled)
        
        # Predecir
        pred = model.predict(X_pca, verbose=0)[0]
        clase = np.argmax(pred)
        prob = pred[clase] * 100
        
        # Mostrar resultado
        st.markdown("---")
        st.subheader("📈 Resultado")
        
        col1, col2, col3 = st.columns(3)
        
        if clase == 0:
            with col1:
                st.error("❌ **MALO**")
            with col2:
                st.metric("Confianza", f"{prob:.1f}%")
            with col3:
                st.error("Alto Riesgo")
            st.error("**NO APROBAR**: Cliente con MAL crédito")
                
        elif clase == 1:
            with col1:
                st.warning("⚠️ **NORMAL**")
            with col2:
                st.metric("Confianza", f"{prob:.1f}%")
            with col3:
                st.warning("Riesgo Moderado")
            st.warning("**APROBACIÓN CONDICIONAL**: Evaluar con cuidado")
            
        else:
            with col1:
                st.success("✅ **BUENO**")
            with col2:
                st.metric("Confianza", f"{prob:.1f}%")
            with col3:
                st.success("Bajo Riesgo")
            st.success("**APROBAR**: Cliente con BUEN crédito")
        
        # Mostrar probabilidades detalladas
        with st.expander("📊 Ver probabilidades detalladas"):
            st.write(f"- Malo (0): {pred[0]*100:.1f}%")
            st.write(f"- Normal (1): {pred[1]*100:.1f}%")
            st.write(f"- Bueno (2): {pred[2]*100:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Error en predicción: {e}")
        st.write("Tipo de error:", type(e).__name__)
        st.write("Detalles para debugging:")
        st.write("DataFrame columns:", input_data.columns.tolist())