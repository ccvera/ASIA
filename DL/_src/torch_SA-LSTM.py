import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load and preprocess the dataset
file_path = '/home/fcsc/ccalvo/MoEvDefinitiva/MoEvDefinitivo/datasets/2008_2018.csv'
#file_path = '/home/fcsc/ccalvo/ML_meteo/utils/SCRIPTS/test/2014-04-01.csv'

df = pd.read_csv(file_path)

# Select features and label
X = df.drop(['DATE','TIMESTAMP','RAINC','RAINNC','RAIN_CHE' ,'RAIN_WRF','RANGE_CHE','RANGE_WRF','RAINING_ERROR','RANGE_ERROR','YEAR','MONTH','RAINING_CHE'], axis=1)
y = df['RAINING_CHE']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the SA-LSTM model
class SelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(SelfAttention, self).__init__()
        self.attention_size = attention_size
        self.attention = nn.Linear(attention_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
    
    def forward(self, lstm_output):
        attention_score = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(self.context_vector(attention_score), dim=1)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

class SALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super(SALSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.self_attention = SelfAttention(attention_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_out = self.self_attention(lstm_out)
        out = self.fc(attn_out)
        return out

# Initialize the model, loss function, and optimizer
input_dim = X_train_scaled.shape[1]
hidden_dim = 64
attention_dim = 64
output_dim = 1
model = SALSTM(input_dim, hidden_dim, attention_dim, output_dim)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 50
batch_size = 64

# Create DataLoader for batching
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(1)  # Add batch dimension
        y_batch = y_batch.unsqueeze(1)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.unsqueeze(1)  # Add batch dimension
    y_pred = model(X_test_tensor)
    y_pred_prob = torch.sigmoid(y_pred).squeeze().numpy()
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    y_true = y_test_tensor.numpy()

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_class)
print(f'Accuracy: {accuracy}')

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_class)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('_images/confusion_matrix_salstm.png')
plt.show()

# Classification report
class_report = classification_report(y_true, y_pred_class)
print('Classification Report:')
print(class_report)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('_images/roc_curve_salstm.png')
plt.show()

# Save the model
torch.save(model.state_dict(), '_modelos/salstm_model.pth')

# Save the scaler
joblib.dump(scaler, '_modelos/scaler_salstm.pkl')
