import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    """Carga todos los archivos necesarios para la predicción"""
    
    # Verificar que los archivos existen
    required_files = {
        'modelo_credito.h5': 'Modelo H5',
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
        # Cargar el modelo H5
        model = load_model('modelo_credito_final.h5', compile=False)
        
        # Recompilar el modelo (necesario para algunas operaciones)
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Cargar los demás archivos
        scaler = joblib.load('minmax_scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        pca = joblib.load('pca_model.joblib')
        
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
        # Mostrar los datos ingresados
        with st.expander("📋 Datos ingresados", expanded=False):
            st.json({
                "Cuentas Bancarias": bank_accounts,
                "Tarjetas Crédito": credit_cards,
                "Tasa Interés": interest_rate,
                "Deuda Pendiente": outstanding_debt,
                "Antigüedad Crédito": credit_history,
                "Mix Crédito": credit_mix,
                "Pago Mínimo": payment_min,
                "Días Retraso": delay_days,
                "Pagos Retrasados": delayed_payments,
                "Consultas Crédito": credit_inquiries
            })
        
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
            0: {"nombre": "MALO 🔴", "desc": "Alto riesgo crediticio - No recomendado", "color": "red"},
            1: {"nombre": "NORMAL 🟡", "desc": "Riesgo crediticio moderado - Evaluar con cuidado", "color": "orange"},
            2: {"nombre": "BUENO 🟢", "desc": "Bajo riesgo crediticio - Recomendado", "color": "green"}
        }
        
        with col1:
            st.metric("Clasificación", etiquetas[clase]["nombre"])
        
        with col2:
            st.metric("Confianza", f"{probabilidades[clase]:.1f}%")
        
        with col3:
            st.metric("Nivel de Riesgo", etiquetas[clase]["desc"])
        
        # Barra de progreso para la probabilidad
        st.progress(int(probabilidades[clase]) / 100)
        
        # Probabilidades detalladas
        st.subheader("📊 Distribución de Probabilidades:")
        prob_df = pd.DataFrame({
            'Clase': ['Malo (0)', 'Normal (1)', 'Bueno (2)'],
            'Probabilidad': probabilidades,
            'Estado': ['🔴' if i == 0 else '🟡' if i == 1 else '🟢' for i in range(3)]
        })
        
        # Mostrar como barras horizontales
        for _, row in prob_df.iterrows():
            st.write(f"{row['Estado']} **{row['Clase']}**: {row['Probabilidad']:.1f}%")
            st.progress(int(row['Probabilidad']) / 100)
        
        # Recomendación final
        st.subheader("💡 Recomendación:")
        if clase == 0:
            st.error("❌ **NO APROBAR**: El cliente presenta alto riesgo crediticio. Se recomienda rechazar la solicitud o requerir garantías significativas.")
        elif clase == 1:
            st.warning("⚠️ **APROBACIÓN CONDICIONAL**: El cliente tiene riesgo moderado. Se recomienda:\n"
                      "- Límite de crédito bajo\n"
                      "- Tasa de interés más alta\n"
                      "- Revisión periódica")
        else:
            st.success("✅ **APROBAR**: El cliente presenta excelente historial crediticio. Se recomienda:\n"
                      "- Límite de crédito alto\n"
                      "- Tasa de interés preferencial\n"
                      "- Ofrecer productos adicionales")
            
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")
        st.write("Detalles del error:", str(e))
        
        # Mostrar información de debug
        with st.expander("🔧 Información de Debug", expanded=False):
            st.write("Tipo de error:", type(e).__name__)
            st.write("Datos procesados:")
            st.write(input_data)

# Footer
st.markdown("---")
st.markdown("💡 *Este sistema utiliza un modelo de red neuronal entrenado con datos históricos para predecir el riesgo crediticio.*")