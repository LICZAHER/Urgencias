import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# Cargar datos desde la base de datos
data = pd.read_csv('Tacotalpa2024.csv')

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), ['SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC'])
    ],
    remainder='passthrough'
)

scaler = StandardScaler()
kmeans = KMeans(n_clusters=5, random_state=42)

# Preprocesar y escalar datos
X_preprocessed = preprocessor.fit_transform(data[['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC']])
X_scaled = scaler.fit_transform(X_preprocessed)
kmeans.fit(X_scaled)

# Predecir el cluster para cada muestra
data['CLUSTER'] = kmeans.predict(X_scaled)

# Guardar el DataFrame con la columna CLUSTER para la interfaz Streamlit
data.to_csv('Tacotalpa2024_with_clusters.csv', index=False)

###################################################################################3
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import joblib

@st.cache_data
def load_data():
    return pd.read_csv('Tacotalpa2024.csv')

@st.cache_resource
def load_model_and_data():
    data = load_data()
    
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        scaler = joblib.load('scaler.joblib')
        kmeans = joblib.load('kmeans.joblib')
        print("Modelos cargados con éxito")
    except FileNotFoundError:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False), ['SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC'])
            ],
            remainder='passthrough'
        )

        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=5, random_state=42)

        X_preprocessed = preprocessor.fit_transform(data[['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC']])
        X_scaled = scaler.fit_transform(X_preprocessed)
        kmeans.fit(X_scaled)
        
        joblib.dump(preprocessor, 'preprocessor.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(kmeans, 'kmeans.joblib')
        print("Modelos guardados con éxito")
        
    X_preprocessed = preprocessor.transform(data[['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC']])
    X_scaled = scaler.transform(X_preprocessed)
    data['CLUSTER'] = kmeans.predict(X_scaled)
    
    return preprocessor, scaler, kmeans, data

preprocessor, scaler, kmeans, data = load_model_and_data()

# Obtener valores únicos para cada opción
sexos = data['SEXO'].unique()
tipos_urgencia = data['TIPOURGENCIA'].unique()
afecciones_principales = data['AFECPRIN'].unique()
afecciones_secundarias = data['AFEC'].unique()

# Definir un rango o lista de edades predefinidas
edades = list(range(0, 101))  # De 0 a 100 años

# Interfaz Streamlit
st.title('Predicción de Medicamentos')

# Recoger inputs del usuario con opciones dinámicas
edad = st.selectbox('Edad ', options=edades)
sexo = st.selectbox('Sexo (1=Masculino, 0=Femenino)', options=sexos)
tipo_urgencia = st.selectbox('Tipo de Urgencia (1= Urgencia Calificana, 0= Urgencia no Calificada)', options=tipos_urgencia)
afecprin = st.selectbox('Afección Principal', options=afecciones_principales)
afec = st.selectbox('Afección Secundaria', options=afecciones_secundarias)

# Botón para realizar la predicción
if st.button('Predecir Medicamentos'):
    try:
        # Preparar los datos para la predicción
        input_data = pd.DataFrame([[edad, sexo, tipo_urgencia, afecprin, afec]],
                                  columns=['EDAD', 'SEXO', 'TIPOURGENCIA', 'AFECPRIN', 'AFEC'])

        input_data_preprocessed = preprocessor.transform(input_data)
        input_data_scaled = scaler.transform(input_data_preprocessed)

        # Predecir el cluster
        cluster = kmeans.predict(input_data_scaled)

        # Filtrar medicamentos y descripciones para el cluster, afección principal y secundaria
        cluster_data = data[
            (data['CLUSTER'] == cluster[0]) &
            (data['AFECPRIN'] == afecprin) &
            (data['AFEC'] == afec)
        ][['MEDICAMENTOS', 'DESCRIPCION']].drop_duplicates()

        # Mostrar resultados en una tabla
        if not cluster_data.empty:
            st.write('Medicamentos recomendados para el paciente:')
            st.dataframe(cluster_data)
        else:
            st.write('No se encontraron medicamentos recomendados para el paciente con la afección principal y secundaria seleccionadas en este cluster.')
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")