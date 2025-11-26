import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objs as go

# Set page config
st.set_page_config(layout="wide", page_title="Apple Stock Prediction")

st.title("🍎 Apple Stock Price Forecast (Next 30 Days)")
st.write("This app uses a Deep Learning (LSTM) model to forecast stock prices.")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Read the CSV file
    # Ensure AAPL.csv is in the same folder or provide correct path
    df = pd.read_csv('AAPL.csv')
    return df

try:
    df = load_data()
    st.subheader("Raw Data")
    st.dataframe(df.head()))
except FileNotFoundError:
    st.error("Error: 'AAPL.csv' file not found. Please upload it to your GitHub repository.")
    st.stop()

# --- 2. Preprocessing ---
# Function to prepare data
def prepare_data(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Select the 'Close' column for prediction
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) # Handle date parsing
df_close = df.reset_index()['Close']

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_close_scaled = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

# Split data (We use the whole dataset for training to predict the FUTURE)
training_size = int(len(df_close_scaled) * 0.70)
test_size = len(df_close_scaled) - training_size
train_data, test_data = df_close_scaled[0:training_size, :], df_close_scaled[training_size:len(df_close_scaled), :1]

# Parameters
time_step = 100 # Look back 100 days

# Create datasets
X_train, y_train = prepare_data(train_data, time_step)
X_test, y_test = prepare_data(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- 3. Build & Train Model ---
# We use st.cache_resource to avoid retraining every time the app reloads
@st.cache_resource
def StreamlitCallback(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train with a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update streamlit progress
    import tensorflow as keras
    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / 10
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Training Model... Epoch {epoch+1}/10")

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1, callbacks=[StreamlitCallback()])
    
    status_text.text("Model Trained Successfully!")
    return model

if st.button("Train Model & Predict"):
    with st.spinner("Training the Deep Learning Model... This might take a minute."):
        model = StreamlitCallback(X_train, y_train)

    # --- 4. Prediction Logic ---
    # Predictions for validation (Test Set)
    test_predict = model.predict(X_test)
    
    # Inverse transform to get actual values
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- 5. Forecast Next 30 Days ---
    # Get the last 100 days of data to start the prediction
    x_input = test_data[len(test_data)-100:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    days_to_predict = 30

    while(i < days_to_predict):
        if(len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    forecast_vals = scaler.inverse_transform(lst_output)
    
    # --- 6. Visualization ---
    
    st.subheader("Forecast Results")
    
    # Create future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    
    # DataFrame for forecast
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': forecast_vals.flatten()})
    st.dataframe(forecast_df)

    # Plotting with Plotly for interactivity
    fig = go.Figure()

    # Original Data (Last 300 days for clarity)
    fig.add_trace(go.Scatter(
        x=df['Date'].iloc[-300:], 
        y=df['Close'].iloc[-300:],
        mode='lines',
        name='Historical Data'
    ))

    # Forecast Data
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=forecast_vals.flatten(),
        mode='lines+markers',
        name='30-Day Forecast',
        line=dict(color='red')
    ))

    fig.update_layout(title="Apple Stock Price: History + 30 Day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Forecasting Complete!")
