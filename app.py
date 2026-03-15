import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Crédito",
    page_icon="🏦",
    layout="wide"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🏦 Clasificador de Valor Crediticio")
st.markdown("""
Esta aplicación predice la categoría de crédito de un solicitante (Malo, Normal o Bueno)
basado en sus datos financieros y crediticios.
""")

# --- CARGA DE MODELOS Y PREPROCESADORES ---
@st.cache_resource
def load_models():
    """Carga todos los artefactos necesarios para la predicción"""
    try:
        # Definir rutas de los archivos
        model_path = "modelo_credito.keras"
        scaler_path = "minmax_scaler.joblib"
        pca_path = "pca_model.joblib"
        encoders_path = "label_encoders.joblib"
        
        # Verificar que existen los archivos
        required_files = [model_path, scaler_path, pca_path, encoders_path]
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"❌ No se encuentran los archivos: {', '.join(missing_files)}")
            st.info("Asegúrate de que los archivos estén en el mismo directorio que app.py")
            return None, None, None, None      
                
        # Desactivar warnings
        tf.get_logger().setLevel('ERROR')
        
        # Definir una versión compatible de InputLayer
        class InputLayerCompat(layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                # Hacer una copia para no modificar el original
                config = config.copy()

                # Convertir batch_shape a batch_input_shape si existe (cambio en TF 2.13)
                if 'batch_shape' in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                
                # Eliminar 'optional' si existe (no existe en TF 2.13)
                if 'optional' in config:
                    del config['optional']
                
                # Eliminar otros argumentos que puedan causar problemas
                if 'ragged' in config and config['ragged'] is False:
                    del config['ragged']  # TF 2.13 no siempre maneja ragged
                
                return super().from_config(config)
        
        # Definir una versión compatible de Dense
        class DenseCompat(layers.Dense):
            @classmethod
            def from_config(cls, config):
                # Eliminar parámetros que no existen en versiones antiguas
                if 'quantization_config' in config:
                    del config['quantization_config']
                return super().from_config(config)
        
        # Registrar las clases personalizadas
        custom_objects = {
            'InputLayer': InputLayerCompat,
            'Dense': DenseCompat
        }
        
        try:
            # Intentar carga con custom_objects
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            st.success("✅ Modelo cargado correctamente (con capas compatibles)")
            
        except Exception as e:
            st.error(f"❌ Error en carga: {str(e)}")
            return None, None, None, None
        
        # Recompilar el modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Cargar los demás artefactos
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        label_encoders = joblib.load(encoders_path)
        
        st.success("✅ Todos los modelos cargados correctamente")
        return model, scaler, pca, label_encoders
        
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {str(e)}")
        return None, None, None, None

# Cargar modelos al iniciar
model, scaler, pca, label_encoders = load_models()

# --- VALORES ORIGINALES DE VARIABLES CATEGÓRICAS (ANTES DEL LABEL ENCODING) ---
# Estos son los valores que el usuario debe seleccionar
credit_mix_options = ['Bad', 'Standard', 'Good']
payment_min_options = ['No', 'Yes']

# Mapeo para mostrar nombres más amigables en español
credit_mix_labels = {
    'Bad': 'Mala',
    'Standard': 'Estándar',
    'Good': 'Buena'
}

payment_min_labels = {
    'No': 'No',
    'Yes': 'Sí'
}

# --- NOMBRES DE LAS CARACTERÍSTICAS EN ESPAÑOL (como las espera el modelo) ---
# Estas son las 10 características seleccionadas por SelectKBest y renombradas en el notebook
FEATURES_ESPAÑOL = [
    'Num_Cuentas_Bancarias',
    'Num_Tarjetas_Credito',
    'Tasa_Interes',
    'Retraso_Desde_Vencimiento',
    'Num_Pagos_Retrasados',
    'Num_Consultas_Credito',
    'Mezcla_Credito',
    'Deuda_Pendiente',
    'Antiguedad_Historial_Crediticio',
    'Pago_Monto_Minimo'
]

# Mapeo de nombres amigables para mostrar en la interfaz
nombres_amigables = {
    'Num_Cuentas_Bancarias': 'Número de Cuentas Bancarias',
    'Num_Tarjetas_Credito': 'Número de Tarjetas de Crédito',
    'Tasa_Interes': 'Tasa de Interés (%)',
    'Retraso_Desde_Vencimiento': 'Retraso desde vencimiento (días)',
    'Num_Pagos_Retrasados': 'Número de Pagos Retrasados',
    'Num_Consultas_Credito': 'Número de Consultas de Crédito',
    'Mezcla_Credito': 'Mezcla de Crédito',
    'Deuda_Pendiente': 'Deuda Pendiente (USD)',
    'Antiguedad_Historial_Crediticio': 'Antigüedad del Historial Crediticio (años)',
    'Pago_Monto_Minimo': 'Pago del Monto Mínimo'
}

# --- INTERFAZ DE USUARIO ---
st.header("📋 Datos del Solicitante")

# Crear columnas para organizar los inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Información Financiera")
    
    num_cuentas_bancarias = st.number_input(
        nombres_amigables['Num_Cuentas_Bancarias'],
        min_value=0, max_value=20, value=3,
        help="Cantidad de cuentas bancarias que posee"
    )
    
    num_tarjetas_credito = st.number_input(
        nombres_amigables['Num_Tarjetas_Credito'],
        min_value=0, max_value=20, value=2,
        help="Cantidad de tarjetas de crédito activas"
    )
    
    tasa_interes = st.number_input(
        nombres_amigables['Tasa_Interes'],
        min_value=0.0, max_value=50.0, value=15.0, step=0.5,
        help="Tasa de interés promedio de sus préstamos"
    )
    
    deuda_pendiente = st.number_input(
        nombres_amigables['Deuda_Pendiente'],
        min_value=0.0, max_value=10000.0, value=1000.0, step=100.0,
        help="Monto total de deuda pendiente"
    )

with col2:
    st.subheader("📅 Historial de Pagos")
    
    retraso_vencimiento = st.number_input(
        nombres_amigables['Retraso_Desde_Vencimiento'],
        min_value=0, max_value=100, value=15,
        help="Promedio de días de retraso en pagos"
    )
    
    num_pagos_retrasados = st.number_input(
        nombres_amigables['Num_Pagos_Retrasados'],
        min_value=0, max_value=50, value=5,
        help="Cantidad total de pagos realizados con retraso"
    )
    
    num_consultas_credito = st.number_input(
        nombres_amigables['Num_Consultas_Credito'],
        min_value=0, max_value=30, value=3,
        help="Cantidad de veces que ha solicitado reportes de crédito"
    )
    
    antiguedad_historial = st.number_input(
        nombres_amigables['Antiguedad_Historial_Crediticio'],
        min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        help="Años desde que inició su historial crediticio"
    )

with col3:
    st.subheader("📊 Variables Categóricas")
    
    mezcla_credito = st.selectbox(
        nombres_amigables['Mezcla_Credito'],
        options=credit_mix_options,
        format_func=lambda x: credit_mix_labels[x],
        help="Tipo de mezcla de crédito: Bad (Mala), Standard (Estándar), Good (Buena)"
    )
    
    pago_monto_minimo = st.selectbox(
        nombres_amigables['Pago_Monto_Minimo'],
        options=payment_min_options,
        format_func=lambda x: payment_min_labels[x],
        help="Indica si paga al menos el monto mínimo requerido"
    )

# --- BOTÓN DE PREDICCIÓN ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predecir Score Crediticio", use_container_width=True)

# --- FUNCIÓN DE PREDICCIÓN ---
def predecir_credito(datos_usuario, model, scaler, pca, label_encoders):
    """
    Función que aplica todo el pipeline de preprocesamiento y realiza la predicción
    """
    try:
        # Crear copia para no modificar el original
        datos_procesados = datos_usuario.copy()
        
        # 1. APLICAR LABEL ENCODING a variables categóricas
        # Convertir a los valores numéricos que espera el modelo
        
        # Para Mezcla_Credito
        if 'Mezcla_Credito' in datos_procesados.columns:
            le_credit_mix = label_encoders['Credit_Mix']  # Nota: en el encoder está con nombre inglés
            valor_original = datos_procesados['Mezcla_Credito'].iloc[0]
            valor_codificado = le_credit_mix.transform([valor_original])[0]
            datos_procesados['Mezcla_Credito'] = valor_codificado
        
        # Para Pago_Monto_Minimo
        if 'Pago_Monto_Minimo' in datos_procesados.columns:
            le_payment = label_encoders['Payment_of_Min_Amount']  # Nota: en el encoder está con nombre inglés
            valor_original = datos_procesados['Pago_Monto_Minimo'].iloc[0]
            valor_codificado = le_payment.transform([valor_original])[0]
            datos_procesados['Pago_Monto_Minimo'] = valor_codificado
        
        # 2. SELECCIONAR las características en el orden correcto
        # El modelo espera exactamente estas 10 características en español
        X = datos_procesados[FEATURES_ESPAÑOL]
        
        # 3. NORMALIZAR con MinMaxScaler
        X_scaled = scaler.transform(X)
        
        # 4. APLICAR PCA (reduce a 8 componentes)
        X_pca = pca.transform(X_scaled)
        
        # 5. REALIZAR PREDICCIÓN
        prediccion_proba = model.predict(X_pca, verbose=0)
        clase_predicha = np.argmax(prediccion_proba, axis=1)[0]
        probabilidades = prediccion_proba[0]
        
        return clase_predicha, probabilidades
        
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None

# --- MOSTRAR RESULTADOS ---
if predict_button:
    if model is not None and scaler is not None and pca is not None and label_encoders is not None:
        
        # Crear DataFrame con los datos ingresados (usando nombres en español)
        datos_usuario = pd.DataFrame({
            'Num_Cuentas_Bancarias': [num_cuentas_bancarias],
            'Num_Tarjetas_Credito': [num_tarjetas_credito],
            'Tasa_Interes': [tasa_interes],
            'Retraso_Desde_Vencimiento': [retraso_vencimiento],
            'Num_Pagos_Retrasados': [num_pagos_retrasados],
            'Num_Consultas_Credito': [num_consultas_credito],
            'Mezcla_Credito': [mezcla_credito],  # Valor original (Bad/Standard/Good)
            'Deuda_Pendiente': [deuda_pendiente],
            'Antiguedad_Historial_Crediticio': [antiguedad_historial],
            'Pago_Monto_Minimo': [pago_monto_minimo]  # Valor original (Yes/No)
        })
        
        # Mostrar resumen de los datos ingresados
        with st.expander("📋 Ver datos ingresados"):
            st.dataframe(datos_usuario, use_container_width=True)
        
        # Realizar predicción
        with st.spinner("Procesando solicitud..."):
            clase, probabilidades = predecir_credito(datos_usuario, model, scaler, pca, label_encoders)
        
        if clase is not None:
            st.markdown("---")
            st.header("✅ Resultado de la Clasificación")
            
            # Mostrar resultado en 3 columnas
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.subheader("📊 Clasificación")
                
                # Definir colores según la clase
                if clase == 0:
                    st.error(f"### **🔴 CRÉDITO MALO**")
                    resultado_texto = "Malo"
                    color_clase = "🔴"
                elif clase == 1:
                    st.warning(f"### **🟡 CRÉDITO NORMAL**")
                    resultado_texto = "Normal"
                    color_clase = "🟡"
                else:  # clase == 2
                    st.success(f"### **🟢 CRÉDITO BUENO**")
                    resultado_texto = "Bueno"
                    color_clase = "🟢"
            
            with col_res2:
                st.subheader("📈 Probabilidades")
                
                # Crear DataFrame con probabilidades
                prob_df = pd.DataFrame({
                    'Categoría': ['Malo (0)', 'Normal (1)', 'Bueno (2)'],
                    'Probabilidad': [f"{probabilidades[0]:.2%}", f"{probabilidades[1]:.2%}", f"{probabilidades[2]:.2%}"]
                })
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            with col_res3:
                st.subheader("📋 Recomendación")
                
                if clase == 0:
                    st.info("""
                    **Recomendación:**  
                    ❌ **No aprobar crédito**  
                    - Alto riesgo de incumplimiento  
                    - Historial crediticio deficiente  
                    - Considere asesoría financiera
                    """)
                elif clase == 1:
                    st.info("""
                    **Recomendación:**  
                    ⚠️ **Evaluar con precaución**  
                    - Riesgo moderado  
                    - Historial con algunas irregularidades  
                    - Considere un monto menor o mayor tasa
                    """)
                else:
                    st.info("""
                    **Recomendación:**  
                    ✅ **Aprobar crédito**  
                    - Bajo riesgo de incumplimiento  
                    - Excelente historial crediticio  
                    - Ofrecer mejores condiciones
                    """)
            
            # Mostrar barra de probabilidad
            st.subheader("Distribución de Probabilidad")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                st.markdown(f"**Malo** {probabilidades[0]:.2%}")
                st.progress(float(probabilidades[0]))
            
            with prob_col2:
                st.markdown(f"**Normal** {probabilidades[1]:.2%}")
                st.progress(float(probabilidades[1]))
            
            with prob_col3:
                st.markdown(f"**Bueno** {probabilidades[2]:.2%}")
                st.progress(float(probabilidades[2]))
            
            # Mostrar nota sobre la confianza
            confianza_max = probabilidades.max()
            st.markdown(f"**Nivel de confianza:** {confianza_max:.2%} en la predicción")
            
    else:
        st.error("No se pudieron cargar los modelos. Verifica que los archivos estén en el directorio correcto.")

# --- INFORMACIÓN ADICIONAL ---
with st.expander("ℹ️ Acerca de esta aplicación"):
    st.markdown("""
    ### Modelo de Clasificación de Crédito
    
    Esta aplicación utiliza una **Red Neuronal Artificial** entrenada con un dataset de solicitantes de crédito.
    
    **Características utilizadas (en español, como espera el modelo):**
    - Num_Cuentas_Bancarias
    - Num_Tarjetas_Credito
    - Tasa_Interes
    - Retraso_Desde_Vencimiento
    - Num_Pagos_Retrasados
    - Num_Consultas_Credito
    - Mezcla_Credito (Bad/Standard/Good)
    - Deuda_Pendiente
    - Antiguedad_Historial_Crediticio
    - Pago_Monto_Minimo (Yes/No)
    
    **Preprocesamiento (aplicado automáticamente):**
    1. Label Encoding para variables categóricas
    2. Selección de características (SelectKBest)
    3. Normalización MinMax
    4. Reducción de dimensionalidad PCA (8 componentes)
    
    **Arquitectura del modelo:**
    - Capa oculta 1: 64 neuronas (ReLU)
    - Capa oculta 2: 32 neuronas (ReLU)
    - Capa oculta 3: 16 neuronas (ReLU)
    - Capa de salida: 3 neuronas (Softmax)
    
    **Métricas del modelo:**
    - Precisión en validación: ~75%
    - Clases: 0 (Malo), 1 (Normal), 2 (Bueno)
    """)