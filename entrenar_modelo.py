import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib

# Cargar el conjunto de datos
ruta_archivo = r'C:\Users\izzim\Desktop\Ismael\Urgencias\Tacotalpa.csv'
df = pd.read_csv(ruta_archivo)

# Definir los codificadores de etiquetas para las variables categóricas
label_encoder_edad = LabelEncoder()
label_encoder_sexo = LabelEncoder()
label_encoder_tipo_urgencia = LabelEncoder()
label_encoder_afec_prin = LabelEncoder()
label_encoder_afec = LabelEncoder()
label_encoder_medicamentos = LabelEncoder()

# Codificar las variables categóricas
df['EDAD'] = label_encoder_edad.fit_transform(df['EDAD'])
df['SEXO'] = label_encoder_sexo.fit_transform(df['SEXO'])
df['TIPOURGENCIA'] = label_encoder_tipo_urgencia.fit_transform(df['TIPOURGENCIA'])
df['AFECPRIN'] = label_encoder_afec_prin.fit_transform(df['AFECPRIN'])
df['AFEC'] = label_encoder_afec.fit_transform(df['AFEC'])
df['MEDICAMENTOS'] = label_encoder_medicamentos.fit_transform(df['MEDICAMENTOS'])

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