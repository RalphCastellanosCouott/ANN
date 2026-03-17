import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    """Carga todos los archivos necesarios para la predicción"""
    
    # Verificar que los archivos existen
    required_files = {
        'modelo_credito.h5': 'Modelo H5 (para pesos)',
        'minmax_scaler.joblib': 'Scaler',
        'label_encoders.joblib': 'Label Encoders',
        'pca_model.joblib': 'PCA'
    }
    
    missing_files = []
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing_files.append(f"{description} ({file})")
    
    if missing_files:
        st.error("❌ Archivos faltantes:")
        for file in missing_files:
            st.error(f"   - {file}")
        st.info(f"Directorio actual: {os.getcwd()}")
        st.info(f"Archivos encontrados: {os.listdir('.')}")
        return None, None, None, None
    
    try:
        # PASO 1: Cargar los archivos auxiliares
        scaler = joblib.load('minmax_scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        pca = joblib.load('pca_model.joblib')
        
        # PASO 2: Reconstruir el modelo manualmente (MISMA ARQUITECTURA que en el notebook)
        # Usamos una función para crear el modelo desde cero
        def create_model():
            input_layer = Input(shape=(8,), name='input_layer')
            x = Dense(64, activation='relu', name='dense')(input_layer)
            x = Dense(32, activation='relu', name='dense_1')(x)
            x = Dense(16, activation='relu', name='dense_2')(x)
            output_layer = Dense(3, activation='softmax', name='dense_3')(x)
            model = Model(inputs=input_layer, outputs=output_layer)
            return model
        
        model = create_model()
        
        # PASO 3: Cargar SOLO los pesos (no toda la arquitectura)
        try:
            # Cargar pesos ignorando la arquitectura
            model.load_weights('modelo_credito.h5')
            st.success("✅ Pesos del modelo cargados correctamente")
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar los pesos: {e}")
            st.info("Usando pesos inicializados aleatoriamente (las predicciones no serán precisas)")
        
        # Compilar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        st.success("✅ Todos los archivos cargados correctamente")
        return model, scaler, label_encoders, pca
        
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None, None, None

# Cargar todos los assets
model, scaler, label_encoders, pca = load_assets()

if model is None:
    st.stop()

st.title("💳 Sistema de Clasificación de Riesgo Crediticio")
st.write("Complete los datos para evaluar al cliente.")

# 2. Formulario - TODOS los nombres en ESPAÑOL
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Información Financiera")
        num_cuentas = st.number_input("Número de Cuentas Bancarias", min_value=0, max_value=20, value=3)
        num_tarjetas = st.number_input("Número de Tarjetas de Crédito", min_value=0, max_value=20, value=2)
        tasa_interes = st.number_input("Tasa de Interés (%)", min_value=0, max_value=50, value=12)
        deuda_pendiente = st.number_input("Deuda Pendiente ($)", min_value=0.0, value=1500.0)
        
        st.subheader("📊 Historial Crediticio")
        antiguedad = st.number_input("Antigüedad del Crédito (años)", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        mezcla_credito = st.selectbox("Mix de Crédito", options=["Good", "Standard", "Bad"])
        pago_minimo = st.selectbox("Pago del Monto Mínimo", options=["Yes", "No"])
    
    with col2:
        st.subheader("⏱️ Historial de Pagos")
        retraso = st.number_input("Días de Retraso", min_value=0, max_value=100, value=5)
        pagos_retrasados = st.number_input("Número de Pagos Retrasados", min_value=0, max_value=50, value=2)
        consultas_credito = st.number_input("Consultas de Crédito Recientes", min_value=0, max_value=30, value=2)
    
    submit = st.form_submit_button("🔍 Analizar Riesgo", use_container_width=True)

if submit:
    try:
        # 1. Crear DataFrame con las features en ESPAÑOL (igual que en el entrenamiento)
        input_data = pd.DataFrame({
            'Num_Cuentas_Bancarias': [float(num_cuentas)],
            'Num_Tarjetas_Credito': [float(num_tarjetas)],
            'Tasa_Interes': [float(tasa_interes)],
            'Retraso_Desde_Vencimiento': [float(retraso)],
            'Num_Pagos_Retrasados': [float(pagos_retrasados)],
            'Num_Consultas_Credito': [float(consultas_credito)],
            'Mezcla_Credito': [mezcla_credito],
            'Deuda_Pendiente': [float(deuda_pendiente)],
            'Antiguedad_Historial_Crediticio': [float(antiguedad)],
            'Pago_Monto_Minimo': [pago_minimo]
        })
        
        # 2. Aplicar LabelEncoder a las variables categóricas
        # Mezcla_Credito
        if 'Mezcla_Credito' in label_encoders:
            input_data['Mezcla_Credito'] = label_encoders['Mezcla_Credito'].transform(input_data['Mezcla_Credito'])
        else:
            mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
            input_data['Mezcla_Credito'] = input_data['Mezcla_Credito'].map(mix_map)
        
        # Pago_Monto_Minimo
        if 'Pago_Monto_Minimo' in label_encoders:
            input_data['Pago_Monto_Minimo'] = label_encoders['Pago_Monto_Minimo'].transform(input_data['Pago_Monto_Minimo'])
        else:
            payment_map = {'No': 0, 'Yes': 1}
            input_data['Pago_Monto_Minimo'] = input_data['Pago_Monto_Minimo'].map(payment_map)
        
        # 3. Aplicar MinMaxScaler
        X_scaled = scaler.transform(input_data)
        
        # 4. Aplicar PCA (8 componentes)
        X_pca = pca.transform(X_scaled)
        
        # 5. Predicción
        prediction = model.predict(X_pca, verbose=0)
        clase = np.argmax(prediction, axis=1)[0]
        probabilidades = prediction[0] * 100
        
        # 6. Mostrar resultados
        st.markdown("---")
        st.subheader("📈 Resultados del Análisis")
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        etiquetas = {
            0: {"nombre": "MALO 🔴", "desc": "Alto riesgo crediticio - No recomendado"},
            1: {"nombre": "NORMAL 🟡", "desc": "Riesgo crediticio moderado - Evaluar con cuidado"},
            2: {"nombre": "BUENO 🟢", "desc": "Bajo riesgo crediticio - Recomendado"}
        }
        
        with col1:
            st.metric("Clasificación", etiquetas[clase]["nombre"])
        
        with col2:
            st.metric("Confianza", f"{probabilidades[clase]:.1f}%")
        
        with col3:
            st.metric("Nivel de Riesgo", etiquetas[clase]["desc"])
        
        # Mostrar probabilidades
        st.subheader("📊 Distribución de Probabilidades:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Malo", f"{probabilidades[0]:.1f}%")
        with col2:
            st.metric("Normal", f"{probabilidades[1]:.1f}%")
        with col3:
            st.metric("Bueno", f"{probabilidades[2]:.1f}%")
        
        # Recomendación
        st.subheader("💡 Recomendación:")
        if clase == 0:
            st.error("❌ **NO APROBAR**: Cliente con MAL crédito")
        elif clase == 1:
            st.warning("⚠️ **APROBACIÓN CONDICIONAL**: Cliente con crédito NORMAL")
        else:
            st.success("✅ **APROBAR**: Cliente con BUEN crédito")
            
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")
        st.write("Detalles del error:", str(e))