import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Streamlit App

st.title("📈 Stock Market Analyzer with ML")
st.write("Upload stock data or fetch from Yahoo Finance and get predictions + visualizations")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., GOOGL, AAPL, TSLA):", "GOOGL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

if st.button("Fetch Data"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found. Try another ticker or date range.")
    else:
        st.success("Data fetched successfully!")
        st.dataframe(df.head())

      
        # Feature Engineering
      
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(window=10).std()
        df["Trend"] = np.where(df["SMA_10"] > df["SMA_20"], 1, 0)
        df.dropna(inplace=True)

      
        # Features & Target
      
        feature_cols = ["Open", "High", "Low", "Volume", "SMA_10", "SMA_20", "Return", "Volatility", "Trend"]
        X = df[feature_cols]
        y = df["Close"]

       # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Visualization: Actual vs Predicted
        st.subheader("📉 Actual vs Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_test.index, y_test, label="Actual", color="blue")
        ax.plot(y_test.index, y_pred, label="Predicted", color="red")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Next-Day Prediction
        st.subheader("🔮 Next Day Price Prediction")
        last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)
        next_day_price = model.predict(last_row)[0]
        st.write(f"Predicted Next Day Close Price: **${next_day_price:.2f}**")

        # Feature Importance
        st.subheader("📌 Feature Importance")
        importance = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance}).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(fi_df["Feature"], fi_df["Importance"], color="skyblue")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # Extra Visualization

        st.subheader("📊 Additional Visualizations")

        # Closing Price History
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close Price", color="blue")
        ax.set_title("Closing Price History")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        # Moving Averages
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close", color="blue")
        ax.plot(df.index, df["SMA_10"], label="SMA 10", color="orange")
        ax.plot(df.index, df["SMA_20"], label="SMA 20", color="green")
        ax.set_title("Stock Price with Moving Averages")
        ax.legend()
        st.pyplot(fig)

        # Return Distribution
        fig, ax = plt.subplots()
        df["Return"].hist(bins=50, ax=ax, color="purple")
        ax.set_title("Return Distribution")
        st.pyplot(fig)
