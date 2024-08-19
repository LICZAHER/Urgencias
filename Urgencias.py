import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib
import streamlit as st

# Cargar el conjunto de datos
ruta_archivo = r'C:\Users\izzim\Desktop\Ismael\Urgencias\Tacotalpa2024.csv'
df = pd.read_csv(ruta_archivo)

# Codificación de variables categóricas
label_encoders = {}
categorical_columns = ['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC', 'MEDICAMENTOS', 'DESCRIPCION']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar características para clustering
X = df[['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC']]

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clusters usando el método del codo
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Elegir el número óptimo de clusters (ajustar según el método del codo)
k_optimo = 5

# Entrenar el modelo K-means
kmeans = KMeans(n_clusters=k_optimo, random_state=42)
kmeans.fit(X_scaled)

# Añadir etiquetas de clusters al dataframe original
df['Cluster'] = kmeans.labels_

# Crear un diccionario para mapear clusters a medicamentos más frecuentes
def get_most_common_medications(cluster):
    cluster_data = df[df['Cluster'] == cluster]
    
    # Obtener los medicamentos más comunes junto con las descripciones
    most_common_medications = {}
    for med_col in ['MEDICAMENTOS']:
        # Obtener los códigos de medicamentos más frecuentes
        med_modes = cluster_data[med_col].mode().tolist()
        # Obtener las descripciones asociadas a esos códigos
        medicamentos = []
        for med_code in med_modes:
            descripcion = cluster_data[cluster_data[med_col] == med_code]['DESCRIPCION'].mode()
            descripcion_text = descripcion.iloc[0] if not descripcion.empty else "Sin descripción"
            medicamento_nombre = label_encoders['MEDICAMENTOS'].inverse_transform([med_code])[0]
            medicamentos.append((medicamento_nombre, descripcion_text))
        most_common_medications[med_col] = medicamentos
    
    return most_common_medications

medicamentos_por_cluster = {cluster: get_most_common_medications(cluster) for cluster in range(k_optimo)}

# Guardar el modelo K-means y el escalador
joblib.dump(kmeans, 'modelo_kmeans.joblib')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(medicamentos_por_cluster, 'medicamentos_por_cluster.pkl')

########################################################################################################

# Cargar el modelo y los codificadores
kmeans = joblib.load('modelo_kmeans.joblib')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
medicamentos_por_cluster = joblib.load('medicamentos_por_cluster.pkl')

# Función para predecir medicamentos
def predecir_medicamentos(edad, sexo, tipo_urgencia, afeccion_principal, afeccion_secundaria):
    try:
        # Codificar las entradas del usuario
        edad_cod = label_encoders['EDAD'].transform([edad])[0]
        sexo_cod = label_encoders['SEXO'].transform([sexo])[0]
        tipo_urgencia_cod = label_encoders['TIPOURGENCIA'].transform([tipo_urgencia])[0]
        afeccion_principal_cod = label_encoders['AFECPRIN'].transform([afeccion_principal])[0]
        afeccion_secundaria_cod = label_encoders['AFEC'].transform([afeccion_secundaria])[0]
        
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
        
        # Obtener medicamentos recomendados para el cluster
        medicamentos_recomendados = medicamentos_por_cluster.get(cluster_pred, {})
        medicamentos = {}
        for med_col in ['MEDICAMENTOS']:
            cod_desc_list = medicamentos_recomendados.get(med_col, [])
            medicamentos[med_col] = cod_desc_list
        
        return medicamentos
    
    except ValueError as ve:
        raise ValueError(f"Error en la codificación de los datos: {ve}")
    except Exception as e:
        raise ValueError(f"Error en el procesamiento de los datos: {e}")

# Interfaz de usuario con Streamlit
st.title("Sistema de Recomendación de Medicamentos")

# Inputs del usuario
edad = st.number_input('Edad', min_value=0, max_value=120, value=30)
sexo = st.selectbox('Sexo', label_encoders['SEXO'].classes_)
tipo_urgencia = st.selectbox('Tipo de urgencia', label_encoders['TIPOURGENCIA'].classes_)
afeccion_principal = st.selectbox('Afección principal', label_encoders['AFECPRIN'].classes_)
afeccion_secundaria = st.selectbox('Afección secundaria', label_encoders['AFEC'].classes_)

if st.button("Predecir medicamentos"):
    try:
        # Validar entradas
        if not (0 <= edad <= 120):
            st.error("La edad debe estar entre 0 y 120 años.")
        else:
            medicamentos_pred = predecir_medicamentos(edad, sexo, tipo_urgencia, afeccion_principal, afeccion_secundaria)
            for med_col, med_list in medicamentos_pred.items():
                st.write(f"{med_col}:")
                if med_list:
                    for nombre, descripcion in med_list:
                        st.write(f"- {nombre}: {descripcion}")
                else:
                    st.write("No hay datos disponibles")
    except ValueError as ve:
        st.error("Ocurrió un error al procesar la solicitud. Por favor, verifica los datos ingresados.")
        st.error(f"Detalles del error: {ve}")
    except Exception as e:
        st.error("Ocurrió un error inesperado.")
        st.error(f"Detalles del error: {e}")