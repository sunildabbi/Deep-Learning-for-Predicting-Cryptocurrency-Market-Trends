import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request
from cryptoviz.util import data_extract, plot_graph
from cryptoviz import flaskapp, eth_model, btc_model, ltc_model, xrp_model

# Method to predict cryptocurrency price and render the plot on home/predict
@flaskapp.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        crypto = request.form.get('crypto')
        start = request.form.get('start')
        end = request.form.get('end')

        if crypto == "bitcoin":
            sym = "BTCUSDT"
        elif crypto == "ethereum":
            sym = "ETHUSDT"
        elif crypto == "ripple":
            sym = "XRPUSDT"
        elif crypto == "litecoin":
            sym = "LTCUSDT"
        else:
            print("Error: Cryptocurrency not available")
            return render_template("home.html", error="Cryptocurrency not available.")

        # Fetch data
        data = data_extract(sym, start, end)

        # Debug: Check if data extraction is successful
        print("Extracted data:", data)
        print("Data shape:", np.shape(data))

        # Check if data is empty
        if data is None or len(data) == 0:
            print("Error: No data received from data_extract()")
            return render_template("home.html", error="No data available for the selected cryptocurrency and date range.")

        # Preprocessing the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        # Reshaping data
        X_test = data[0:len(data) - 1]
        y_test = data[1:len(data)]
        X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

        # Check if test data exists
        if len(X_test) == 0:
            print("Error: No test data available for prediction")
            return render_template("home.html", error="Insufficient data for prediction.")

        # Perform predictions on test data
        if crypto == "bitcoin":
            predicted_price = btc_model.predict(X_test)
        elif crypto == "ethereum":
            predicted_price = eth_model.predict(X_test)
        elif crypto == "ripple":
            predicted_price = xrp_model.predict(X_test)
        else:
            predicted_price = ltc_model.predict(X_test)

        # Convert predictions back to original scale
        predicted_price = scaler.inverse_transform(predicted_price)
        real_price = scaler.inverse_transform(y_test)

        # Plot graph
        p_url = plot_graph(crypto, predicted_price, real_price)

        return render_template("predict.html", plot_url=f'data:image/png;base64,{p_url}')

# Home page that is rendered for every web call
@flaskapp.route("/")
def home():
    return render_template("home.html")
