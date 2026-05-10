import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Load Dataset
df = pd.read_csv("Google_Stock_Price.csv", thousands=',')

# Take Open column and clean data
data = pd.to_numeric(df['Open'], errors='coerce').dropna().values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Split dataset into train and test
train_size = int(len(data_scaled) * 0.8)

train_data = data_scaled[:train_size]
test_data = data_scaled[train_size-60:]   # include previous 60 values

# Create dataset function
def create_dataset(dataset):
    X = []
    y = []

    for i in range(60, len(dataset)):
        X.append(dataset[i-60:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)

# Prepare train and test data
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build RNN Model
model = Sequential()

model.add(SimpleRNN(50, return_sequences=True, input_shape=(60,1)))
model.add(SimpleRNN(50))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prediction
predicted = model.predict(X_test)

# Convert back to original values
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot results
plt.plot(real, color='red', label='Real Price')
plt.plot(predicted, color='blue', label='Predicted Price')

plt.title("Google Stock Price Prediction (RNN)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()