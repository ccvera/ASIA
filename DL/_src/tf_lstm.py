import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar el dataset
#file_path = '/home/fcsc/ccalvo/ML_meteo/utils/SCRIPTS/test/enero/buenos/semana.csv'
file_path = '/home/fcsc/ccalvo/MoEvDefinitiva/MoEvDefinitivo/datasets/2008_2018.csv'

data = pd.read_csv(file_path)

data = data.drop(['DATE','TIMESTAMP','RAINC','RAINNC','RAIN_CHE' ,'RAIN_WRF','RANGE_CHE','RANGE_WRF','RAINING_ERROR','RANGE_ERROR','YEAR','MONTH'], axis=1)

# Paso 2: Preprocesamiento de los datos
# Convertimos las etiquetas categóricas en numéricas si es necesario
if data['RAINING_CHE'].dtype == 'object':
    data['RAINING_CHE'] = data['RAINING_CHE'].astype('category').cat.codes

# Seleccionamos las características y la etiqueta
features = data.drop('RAINING_CHE', axis=1).values
labels = data['RAINING_CHE'].values

# Normalización de las características
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Convertimos las características a secuencias de tiempo (ej. 10 pasos de tiempo)
def create_sequences(features, labels, time_steps=10):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(features_scaled, labels, time_steps)

# Paso 3: Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Construir el modelo LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 5: Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Paso 6: Evaluación
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculamos la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calcular y mostrar las métricas de clasificación: precision, recall, f1-score con 4 decimales
report = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'], digits=4)
print(report)

# Generamos la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Graficar la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Guardar la gráfica
plt.savefig('_images/confusion_matrix_LSTM.png')

plt.show()

# Guardar el modelo entrenado
model.save('_modelos/lstm_model.h5')

print("La matriz de confusión se ha guardado como 'confusion_matrix.png'.")
print("El modelo entrenado se ha guardado como 'lstm_model.h5'.")
