import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    """Carga todos los archivos necesarios para la predicción"""
    try:
        model = load_model('modelo_credito.keras')  # Cambiado a .keras
        scaler = joblib.load('minmax_scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        pca = joblib.load('pca_model.joblib')
        
        # Las features seleccionadas (10 features según tu notebook)
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
        
        return model, scaler, label_encoders, pca, selected_features, column_map
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        st.info("Revisa que los archivos .joblib y .keras estén en el directorio correcto.")
        return None, None, None, None, None, None

# Carga de seguridad
model, scaler, label_encoders, pca, selected_features, column_map = load_assets()

if model is None:
    st.stop()

st.title("💳 Sistema de Clasificación de Riesgo Crediticio")
st.write("Complete los datos para evaluar al cliente.")

# 2. Formulario
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Personales y Financieros")
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Ocupación", 
                                 options=["Doctor", "Engineer", "Lawyer", "Mechanic", "Media_Manager", "Others"])
        annual_income = st.number_input("Ingreso Anual", min_value=0.0, value=50000.0)
        monthly_salary = st.number_input("Salario Mensual Neto", min_value=0.0, value=4000.0)
        
        st.subheader("Información Bancaria")
        bank_accounts = st.number_input("Número de Cuentas Bancarias", min_value=0, max_value=20, value=3)
        credit_cards = st.number_input("Número de Tarjetas de Crédito", min_value=0, max_value=20, value=2)
        interest_rate = st.number_input("Tasa de Interés (%)", min_value=0, max_value=50, value=12)
        num_loans = st.number_input("Número de Préstamos", min_value=0, max_value=20, value=1)
    
    with col2:
        st.subheader("Historial de Pagos")
        delay_days = st.number_input("Días de Retraso", min_value=0, max_value=100, value=5)
        delayed_payments = st.number_input("Número de Pagos Retrasados", min_value=0, max_value=50, value=2)
        credit_inquiries = st.number_input("Consultas de Crédito Recientes", min_value=0, max_value=30, value=2)
        
        st.subheader("Deudas y Crédito")
        credit_mix = st.selectbox("Mix de Crédito", options=["Good", "Standard", "Bad"])
        outstanding_debt = st.number_input("Deuda Pendiente", min_value=0.0, value=1500.0)
        credit_history = st.number_input("Antigüedad Historial Crediticio (años)", min_value=0.0, max_value=40.0, value=5.0)
        payment_min = st.selectbox("Pago del Monto Mínimo", options=["Yes", "No"])
    
    submit = st.form_submit_button("Analizar Riesgo")

if submit:
    try:
        # 1. Crear DataFrame con las features seleccionadas (inglés)
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
            # Mapeo manual si no está el encoder
            mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
            input_data['Credit_Mix'] = input_data['Credit_Mix'].map(mix_map)
        
        # Payment_of_Min_Amount
        if 'Payment_of_Min_Amount' in label_encoders:
            input_data['Payment_of_Min_Amount'] = label_encoders['Payment_of_Min_Amount'].transform(input_data['Payment_of_Min_Amount'])
        else:
            payment_map = {'No': 0, 'Yes': 1}
            input_data['Payment_of_Min_Amount'] = input_data['Payment_of_Min_Amount'].map(payment_map)
        
        # 3. Renombrar a español para mostrar (opcional, para debug)
        input_data_espanol = input_data.rename(columns=column_map)
        
        st.write("📊 Datos ingresados:")
        st.dataframe(input_data_espanol)
        
        # 4. Aplicar MinMaxScaler
        X_scaled = scaler.transform(input_data)
        
        # 5. Aplicar PCA (8 componentes)
        X_pca = pca.transform(X_scaled)
        
        # 6. Predicción
        prediction = model.predict(X_pca, verbose=0)
        clase = np.argmax(prediction, axis=1)[0]
        probabilidades = prediction[0] * 100
        
        # 7. Mostrar resultados
        st.markdown("---")
        st.subheader("📈 Resultados del Análisis")
        
        col1, col2, col3 = st.columns(3)
        
        etiquetas = {
            0: {"nombre": "MALO 🔴", "desc": "Alto riesgo crediticio"},
            1: {"nombre": "NORMAL 🟡", "desc": "Riesgo crediticio moderado"},
            2: {"nombre": "BUENO 🟢", "desc": "Bajo riesgo crediticio"}
        }
        
        with col1:
            st.metric("Clasificación", etiquetas[clase]["nombre"])
        
        with col2:
            st.metric("Confianza", f"{probabilidades[clase]:.1f}%")
        
        with col3:
            st.metric("Descripción", etiquetas[clase]["desc"])
        
        # 8. Mostrar probabilidades detalladas
        st.subheader("📊 Probabilidades por clase:")
        prob_df = pd.DataFrame({
            'Clase': ['Malo (0)', 'Normal (1)', 'Bueno (2)'],
            'Probabilidad': [f"{probabilidades[0]:.1f}%", 
                           f"{probabilidades[1]:.1f}%", 
                           f"{probabilidades[2]:.1f}%"]
        })
        st.dataframe(prob_df, use_container_width=True)
        
        # 9. Recomendación
        st.subheader("💡 Recomendación:")
        if clase == 0:
            st.error("❌ Cliente con MAL crédito. Se recomienda NO otorgar crédito o requerir garantías adicionales.")
        elif clase == 1:
            st.warning("⚠️ Cliente con crédito NORMAL. Se recomienda evaluación adicional y posiblemente un límite de crédito moderado.")
        else:
            st.success("✅ Cliente con BUEN crédito. Se recomienda otorgar crédito con condiciones favorables.")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.write("Detalles del error:", str(e))
        st.write("Tipo de error:", type(e).__name__)
        
        # Debug info
        st.write("Debug - Datos ingresados:")
        st.write(input_data)