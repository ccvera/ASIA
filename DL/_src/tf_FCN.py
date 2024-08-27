import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Cargar el dataset
#file_path = '/home/fcsc/ccalvo/MoEvDefinitiva/MoEvDefinitivo/datasets/2008_2018.csv'
file_path = '/home/fcsc/ccalvo/ML_meteo/utils/SCRIPTS/test/2014-04-01.csv'

data = pd.read_csv(file_path)

# Definir la columna objetivo y las características
target = 'RAINING_CHE'
features = data.columns.drop(['DATE','TIMESTAMP','RAINC','RAINNC','RAIN_CHE' ,'RAIN_WRF','RANGE_CHE','RANGE_WRF','RAINING_ERROR','RANGE_ERROR','YEAR','MONTH','RAINING_CHE'])  # Excluir la columna objetivo y no numéricas

# Separar características y etiqueta
X = data[features].values
y = data[target].values

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Redimensionar los datos para que sean compatibles con la FCN (samples, timesteps, features)
# Aquí asumimos que timesteps=1 (secuencia de un solo paso de tiempo)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Construir el modelo FCN
model = Sequential()
model.add(Conv1D(128, kernel_size=8, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(256, kernel_size=5, activation='relu'))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))  # Salida binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# Evaluar el modelo
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Calcular y mostrar la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calcular y mostrar las métricas de clasificación: precision, recall, f1-score con 4 decimales
report = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'], digits=4)
print(report)

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Rain', 'Rain'])
disp.plot(cmap=plt.cm.Blues)

# Guardar la matriz de confusión como imagen
plt.savefig('_images/confusion_matrix_fcnTF.png')

# Calcular la curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Generar y guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('_images/roc_curve_fcnTF.png')

# Mostrar la curva ROC
plt.show()

# Guardar el modelo
model.save('_modelos/fcnTF_model.h5')
