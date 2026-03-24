# 📈 Currency Forecasting with LSTM

This project is a Deep Learning application developed to predict **USD/TRY** exchange rates. It bridges **Economics** and **Computer Science** by analyzing historical financial data to forecast future price movements.

## 🚀 Overview
- **Objective:** Predicting the next 30 days of USD/TRY exchange rates.
- **Model:** A 2-layer **LSTM (Long Short-Term Memory)** neural network.
- **Data Source:** Yahoo Finance API (`yfinance`).
- **Target Audience:** Exchange students and investors (Core engine for "The Seoul Exchanger" app).

## 📊 Results & Performance
- **RMSE (Root Mean Squared Error):** 0.34 TL
- **Trend Analysis:** The model successfully captured the overall upward trend of the currency with high precision.

![Model Performance](./results/performance_plot.png)

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Preprocessing:** Scikit-learn (MinMaxScaler)

## 🏗️ How to Run
1. Clone the repository: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python main.py`

## 🔮 Future Work: "The Seoul Exchanger"
This model will be integrated into a **Streamlit** web application to provide real-time "Buy/Sell/Wait" decisions for USD, KRW, and Gold/Silver investments, specifically tailored for students in South Korea.
