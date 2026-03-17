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
        return None, None, None, None, None, None
    
    try:
        # PASO 1: Cargar los archivos auxiliares
        scaler = joblib.load('minmax_scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        pca = joblib.load('pca_model.joblib')
        
        # PASO 2: Reconstruir el modelo manualmente (MISMA ARQUITECTURA que en el notebook)
        # Input layer con 8 features (salida del PCA)
        input_layer = Input(shape=(8,), name='input_layer')
        
        # Capas ocultas
        x = Dense(64, activation='relu', name='dense')(input_layer)
        x = Dense(32, activation='relu', name='dense_1')(x)
        x = Dense(16, activation='relu', name='dense_2')(x)
        
        # Capa de salida con 3 neuronas (softmax para clasificación)
        output_layer = Dense(3, activation='softmax', name='dense_3')(x)
        
        # Crear el modelo
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # PASO 3: Cargar SOLO los pesos (no toda la arquitectura)
        try:
            model.load_weights('modelo_credito.h5')
            st.success("✅ Pesos del modelo cargados correctamente")
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar los pesos: {e}")
            st.info("Usando pesos inicializados aleatoriamente (las predicciones no serán precisas)")
        
        # Compilar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Las features seleccionadas (en el orden correcto)
        selected_features = [
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
            'Credit_Mix', 'Outstanding_Debt', 'Credit_History_Age',
            'Payment_of_Min_Amount'
        ]
        
        # Mapa de columnas (inglés a español para mostrar)
        column_map = {
            'Num_Bank_Accounts': 'Num_Cuentas_Bancarias',
            'Num_Credit_Card': 'Num_Tarjetas_Credito',
            'Interest_Rate': 'Tasa_Interes',
            'Delay_from_due_date': 'Retraso_Desde_Vencimiento',
            'Num_of_Delayed_Payment': 'Num_Pagos_Retrasados',
            'Num_Credit_Inquiries': 'Num_Consultas_Credito',
            'Credit_Mix': 'Mezcla_Credito',
            'Outstanding_Debt': 'Deuda_Pendiente',
            'Credit_History_Age': 'Antiguedad_Historial_Crediticio',
            'Payment_of_Min_Amount': 'Pago_Monto_Minimo'
        }
        
        st.success("✅ Todos los archivos cargados correctamente")
        return model, scaler, label_encoders, pca, selected_features, column_map
        
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None, None, None, None, None

# Cargar todos los assets
model, scaler, label_encoders, pca, selected_features, column_map = load_assets()

if model is None:
    st.stop()

st.title("💳 Sistema de Clasificación de Riesgo Crediticio")
st.write("Complete los datos para evaluar al cliente.")

# 2. Formulario
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Información Financiera")
        bank_accounts = st.number_input("Número de Cuentas Bancarias", min_value=0, max_value=20, value=3)
        credit_cards = st.number_input("Número de Tarjetas de Crédito", min_value=0, max_value=20, value=2)
        interest_rate = st.number_input("Tasa de Interés (%)", min_value=0, max_value=50, value=12)
        outstanding_debt = st.number_input("Deuda Pendiente ($)", min_value=0.0, value=1500.0)
        
        st.subheader("📊 Historial Crediticio")
        credit_history = st.number_input("Antigüedad del Crédito (años)", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        credit_mix = st.selectbox("Mix de Crédito", options=["Good", "Standard", "Bad"])
        payment_min = st.selectbox("Pago del Monto Mínimo", options=["Yes", "No"])
    
    with col2:
        st.subheader("⏱️ Historial de Pagos")
        delay_days = st.number_input("Días de Retraso", min_value=0, max_value=100, value=5)
        delayed_payments = st.number_input("Número de Pagos Retrasados", min_value=0, max_value=50, value=2)
        credit_inquiries = st.number_input("Consultas de Crédito Recientes", min_value=0, max_value=30, value=2)
    
    submit = st.form_submit_button("🔍 Analizar Riesgo", use_container_width=True)

if submit:
    try:
        # 1. Crear DataFrame con las features seleccionadas
        input_data = pd.DataFrame({
            'Num_Bank_Accounts': [float(bank_accounts)],
            'Num_Credit_Card': [float(credit_cards)],
            'Interest_Rate': [float(interest_rate)],
            'Delay_from_due_date': [float(delay_days)],
            'Num_of_Delayed_Payment': [float(delayed_payments)],
            'Num_Credit_Inquiries': [float(credit_inquiries)],
            'Credit_Mix': [credit_mix],
            'Outstanding_Debt': [float(outstanding_debt)],
            'Credit_History_Age': [float(credit_history)],
            'Payment_of_Min_Amount': [payment_min]
        })
        
        # 2. Aplicar LabelEncoder a las variables categóricas
        # Credit_Mix
        if 'Credit_Mix' in label_encoders:
            input_data['Credit_Mix'] = label_encoders['Credit_Mix'].transform(input_data['Credit_Mix'])
        else:
            mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
            input_data['Credit_Mix'] = input_data['Credit_Mix'].map(mix_map)
        
        # Payment_of_Min_Amount
        if 'Payment_of_Min_Amount' in label_encoders:
            input_data['Payment_of_Min_Amount'] = label_encoders['Payment_of_Min_Amount'].transform(input_data['Payment_of_Min_Amount'])
        else:
            payment_map = {'No': 0, 'Yes': 1}
            input_data['Payment_of_Min_Amount'] = input_data['Payment_of_Min_Amount'].map(payment_map)
        
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
            st.metric("Malo", f"{probabilidades[0]:.1f}%", delta_color="inverse")
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