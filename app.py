import numpy as np
import torch
import yfinance as yf

from datetime import date, timedelta
from giza.agents.model import GizaModel
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, jsonify

app = Flask(__name__)

# Fetch Historical Price Data
def fetch_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    return data['Close'].values.reshape(-1, 1)

# Preprocess Data
def preprocess_data(data, window_size):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    last_window = data[-window_size:]
    last_window_scaled = scaler.transform(last_window)
    last_window_scaled = torch.tensor(last_window_scaled, dtype=torch.float32).view(1, -1)   

    last_window_scaled = np.array(last_window_scaled).reshape(1, -1)
    return last_window_scaled, scaler


# Make predictions
def predict():
    """
    Predict the ETH price.

    Args:
    last_window_scaled (np.ndarray): Input to the model.

    Returns:
        float: Predicted ETH price.
        string: Request ID.  
    """

    # Parameters
    ticker = "ETH-USD"
    start_date = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
    window_size = 60
    model_id = 784
    version_id = 1
    
    # Fetch and preprocess data
    print("Fetching last 60 trading sessions data")
    data = fetch_data(ticker, start_date)
    (last_window_scaled,scaler) = preprocess_data(data, window_size)

    # Create the model and account
    print("Creating the model instance")
    model = GizaModel(
        id=model_id,
        version=version_id
    )

    # Predict the price
    print("Predicting the price")
    (result, request_id) = model.predict(
        input_feed={"last_window_scaled" :last_window_scaled}, verifiable=True, dry_run=True
    )

    # convert result to tensor  
    tensor_result = torch.tensor(result)
    predicted_price = scaler.inverse_transform(tensor_result.numpy().reshape(-1, 1))
    eth_price_prediction = predicted_price[0,0]
    print(f"Predicted ETH price: {eth_price_prediction}")

    return eth_price_prediction, request_id

@app.route('/')
def hello():
    (eth_price_prediction, request_id) = predict()
    return jsonify(message="Received prediction from server!", eth_price_prediction=eth_price_prediction, request_id=request_id)

if __name__ == '__main__':
    app.run(debug=True)
