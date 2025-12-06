import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

#load ECG5000 dataset
url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
#url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ecg5000.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
df.head()

#preprocess data
# last Column = Label, other columns  = ECG Time-Series values
labels = df.iloc[:, -1]
data = df.iloc[:,  0:-1]

#normalize time-series values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

#binary labels: 0 = normal, 1 = anomaly
y = (labels != 1).astype(int)

#reshape for LSTM input [samples, timesteps, features]
X = np.expand_dims(data_scaled, axis=2)
print("LSTM input shape:", X.shape)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#use only normal data for training
X_train_normal = X_train[y_train == 0]
print(f"Training samples (normal only): {X_train_normal.shape[0]}")

#build LSTM autoencoder
timesteps = X_train_normal.shape[1]
features = X_train_normal.shape[2]

inputs = Input(shape=(timesteps, features))
encoded = LSTM(64, activation='relu', return_sequences=False,
               activity_regularizer=regularizers.l1(1e-5))(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(1, activation='sigmoid', return_sequences=True)(decoded)

lstm_autoencoder = Model(inputs, decoded)
lstm_autoencoder.compile(optimizer='adam', loss='mse')
lstm_autoencoder.summary()

#train autoencoder
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm_autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

#reconstruction errors
X_test_pred = lstm_autoencoder.predict(X_test)
reconstruction_error = np.mean(np.mean(np.square(X_test - X_test_pred), axis=1), axis=1)

X_train_pred = lstm_autoencoder.predict(X_train_normal)
train_error = np.mean(np.mean(np.square(X_train_normal - X_train_pred), axis=1), axis=1)
threshold = np.percentile(train_error, 99)
print(f"Reconstruction Error Threshold: {threshold:.6f}")

#predict anomalies
y_pred = (reconstruction_error > threshold).astype(int)

#evaluate
from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#reconstruction error
plt.figure(figsize=(8,5))
sns.histplot(train_error, bins=50, label='Train (Normal)', color='blue', alpha=0.6)
sns.histplot(reconstruction_error[y_test==0], bins=50, label='Test Normal', color='green', alpha=0.6)
sns.histplot(reconstruction_error[y_test==1], bins=50, label='Test Anomaly', color='red', alpha=0.6)
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.show()

#sample reconstruction
idx = np.random.randint(0, len(X_test))
plt.figure(figsize=(10,5))
plt.plot(X_test[idx].flatten(), label="Original")
plt.plot(X_test_pred[idx].flatten(), label="Reconstructed")
plt.title(f"Sample Reconstruction (True Label: {'Anomaly' if y_test.values[idx]==1 else 'Normal'})")
plt.xlabel("Time Step")
plt.ylabel("Signal Value")
plt.legend()
plt.show()
