import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
X = data.drop('RAINING_CHE', axis=1).values
y = data['RAINING_CHE'].values

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Redimensionar los datos para que sean compatibles con CNN (samples, timesteps, features)
# Aquí asumimos que timesteps=1 (secuencia de un solo paso de tiempo)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Construir el modelo CNN
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Salida binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluar el modelo
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calcular y mostrar la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calcular y mostrar las métricas de clasificación: precision, recall, f1-score
report = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'], digits=4)
print(report)

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Rain', 'Rain'])
disp.plot(cmap=plt.cm.Blues)

# Guardar la matriz de confusión como imagen
plt.savefig('_images/confusion_matrix_CNN.png')

# Guardar el modelo
model.save('_modelos/cnn_model.h5')

# Mostrar la matriz de confusión
plt.show()
