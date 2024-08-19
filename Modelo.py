import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib
import streamlit as st

# Cargar el conjunto de datos
ruta_archivo = r'C:\Users\izzim\Desktop\Ismael\Urgencias\Tacotalpa2024.csv'
df = pd.read_csv(ruta_archivo)

# Definir los codificadores de etiquetas para las variables categóricas
label_encoder_edad = LabelEncoder()
label_encoder_sexo = LabelEncoder()
label_encoder_tipo_urgencia = LabelEncoder()
label_encoder_afec_prin = LabelEncoder()
label_encoder_afec = LabelEncoder()
label_encoder_medicamentos = LabelEncoder()
label_encoder_descripcion = LabelEncoder()

# Codificar las variables categóricas
df['EDAD'] = label_encoder_edad.fit_transform(df['EDAD'])
df['SEXO'] = label_encoder_sexo.fit_transform(df['SEXO'])
df['TIPOURGENCIA'] = label_encoder_tipo_urgencia.fit_transform(df['TIPOURGENCIA'])
df['AFECPRIN'] = label_encoder_afec_prin.fit_transform(df['AFECPRIN'])
df['AFEC'] = label_encoder_afec.fit_transform(df['AFEC'])
df['MEDICAMENTOS'] = label_encoder_medicamentos.fit_transform(df['MEDICAMENTOS'])
df['DESCRIPCION'] = label_encoder_medicamentos.fit_transform(df['DESCRIPCION'])

# Separar características y variable objetivo
X = df[['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC']]
y = df['MEDICAMENTOS']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clusters usando el método del codo
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Elegir un número óptimo de clusters (esto es un ejemplo, ajusta según el método del codo)
k_optimo = 5

# Entrenar el modelo K-means
kmeans = KMeans(n_clusters=k_optimo, random_state=42)
kmeans.fit(X_scaled)

# Añadir etiquetas de clusters al dataframe original
df['Cluster'] = kmeans.labels_

# Guardar los datos clusterizados en un archivo CSV
df.to_csv('datos_pacientes_clusterizados.csv', index=False)

# Crear un diccionario para mapear clusters a medicamentos
medicamentos_por_cluster = df.groupby('Cluster')['MEDICAMENTOS'].agg(lambda x: x.mode()[0]).to_dict()

# Guardar el modelo K-means y el escalador
joblib.dump(kmeans, 'modelo_kmeans.joblib')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_edad, 'label_encoder_edad.pkl')
joblib.dump(label_encoder_sexo, 'label_encoder_sexo.pkl')
joblib.dump(label_encoder_tipo_urgencia, 'label_encoder_tipo_urgencia.pkl')
joblib.dump(label_encoder_afec_prin, 'label_encoder_afec_prin.pkl')
joblib.dump(label_encoder_afec, 'label_encoder_afec.pkl')
joblib.dump(label_encoder_medicamentos, 'label_encoder_medicamentos.pkl')
joblib.dump(medicamentos_por_cluster, 'medicamentos_por_cluster.pkl')
#joblib.dump(medicamentos_por_cluster, 'medicamentos_por_cluster.pkl')

########################################################################################################

# Cargar el modelo y los codificadores
kmeans = joblib.load('modelo_kmeans.joblib')
scaler = joblib.load('scaler.pkl')
label_encoder_edad = joblib.load('label_encoder_edad.pkl')
label_encoder_sexo = joblib.load('label_encoder_sexo.pkl')
label_encoder_tipo_urgencia = joblib.load('label_encoder_tipo_urgencia.pkl')
label_encoder_afec_prin = joblib.load('label_encoder_afec_prin.pkl')
label_encoder_afec = joblib.load('label_encoder_afec.pkl')
label_encoder_medicamentos = joblib.load('label_encoder_medicamentos.pkl')
medicamentos_por_cluster = joblib.load('medicamentos_por_cluster.pkl')

# Función para predecir el medicamento
def predecir_medicamento(edad, sexo, tipo_urgencia, afeccion_principal, afeccion_secundaria):
    try:
        # Convertir entradas a valores codificados
        edad_cod = label_encoder_edad.transform([edad])[0]
        sexo_cod = label_encoder_sexo.transform([sexo])[0]
        tipo_urgencia_cod = label_encoder_tipo_urgencia.transform([tipo_urgencia])[0]
        afeccion_principal_cod = label_encoder_afec_prin.transform([afeccion_principal])[0]
        afeccion_secundaria_cod = label_encoder_afec.transform([afeccion_secundaria])[0]
        
        # Crear un DataFrame para el paciente
        paciente_data = pd.DataFrame({
            'EDAD': [edad_cod],
            'SEXO': [sexo_cod],
            'TIPOURGENCIA': [tipo_urgencia_cod],
            'AFECPRIN': [afeccion_principal_cod],
            'AFEC': [afeccion_secundaria_cod]
        })
        
        # Escalar los datos del paciente
        paciente_data_scaled = scaler.transform(paciente_data)
        
        # Predecir el cluster
        cluster_pred = kmeans.predict(paciente_data_scaled)[0]
        
        # Obtener el medicamento recomendado para el cluster
        medicamento_cod = medicamentos_por_cluster.get(cluster_pred, None)
        medicamento = label_encoder_medicamentos.inverse_transform([medicamento_cod])[0] if medicamento_cod is not None else "Medicamento desconocido"
        return medicamento
    
    except ValueError as ve:
        raise ValueError(f"Error en la codificación de los datos: {ve}")
    except Exception as e:
        raise ValueError(f"Error en el procesamiento de los datos: {e}")

# Interfaz de usuario con Streamlit
st.title("Sistema de Recomendación de Medicamentos")

# Inputs del usuario
edad = st.number_input('Edad', min_value=0, max_value=120, value=30)
sexo = st.selectbox('Sexo', label_encoder_sexo.classes_)
tipo_urgencia = st.selectbox('Tipo de urgencia', label_encoder_tipo_urgencia.classes_)
afeccion_principal = st.selectbox('Afección principal', label_encoder_afec_prin.classes_)
afeccion_secundaria = st.selectbox('Afección secundaria', label_encoder_afec.classes_)

if st.button("Predecir medicamento"):
    try:
        # Validar entradas
        if not (0 <= edad <= 120):
            st.error("La edad debe estar entre 0 y 120 años.")
        else:
            medicamento_pred = predecir_medicamento(edad, sexo, tipo_urgencia, afeccion_principal, afeccion_secundaria)
            st.success(f"Se recomienda el medicamento: **{medicamento_pred}**")
    except ValueError as ve:
        st.error("Ocurrió un error al procesar la solicitud. Por favor, verifica los datos ingresados.")
        st.error(f"Detalles del error: {ve}")
    except Exception as e:
        st.error("Ocurrió un error inesperado.")
        st.error(f"Detalles del error: {e}")
