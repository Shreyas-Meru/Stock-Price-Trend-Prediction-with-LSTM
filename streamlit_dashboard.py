import streamlit as st
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import ta

st.set_page_config(layout="wide", page_title="Stock Price Trend Prediction", page_icon="📈")
st.title("📈 Stock Price Trend Prediction")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
days_to_predict = st.slider("Select Number of Days to Predict", 1, 30, 5)

model = None  # Initialize model
try:
    model = load_model('model_weights.h5')  # Load your model
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    
scaler = MinMaxScaler()

if ticker:
    try:
        with st.spinner('Loading stock data...'):
            df = yf.download(ticker, period='2y')
            if df.empty:
                raise ValueError(f"No data available for {ticker}")
            df.columns = df.columns.droplevel(1)
            
            # Display basic stock info
            latest_prices = df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Open", f"${latest_prices['Open']:.2f}")
            with col2:
                st.metric("High", f"${latest_prices['High']:.2f}")
            with col3:
                st.metric("Low", f"${latest_prices['Low']:.2f}")
            with col4:
                st.metric("Close", f"${latest_prices['Close']:.2f}")
            with col5:
                st.metric("Volume", f"{int(latest_prices['Volume']):,}")

            # Price charts
            st.subheader("📊 Stock Price History")
            st.line_chart(df[['Open', 'High', 'Low', 'Close']])
            
            # Volume chart
            st.subheader("📊 Trading Volume")
            st.bar_chart(df['Volume'])
            
            # Technical indicators
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()

            st.subheader("📉 Closing Price with 50-Day MA")
            st.line_chart(df[['Close', 'MA_50']])
            
            st.subheader("📈 Relative Strength Index (RSI)")
            st.line_chart(df['RSI'])
            
            st.subheader("📈 MACD with Signal Line")
            st.line_chart(df[['MACD', 'MACD_Signal']])
            
            # Prepare data for prediction if model is loaded
            if model:
                prices = df['Close'].values.reshape(-1, 1)
                scaled_prices = scaler.fit_transform(prices)
                window_size = 60  # Adjust based on your model's input requirements
                X_pred = scaled_prices[-window_size:].reshape(1, window_size, 1)

                # Make predictions for multiple days starting from day+2
                future_predictions = []
                current_X = X_pred.copy()

                for _ in range(days_to_predict):
                    predicted_price = model.predict(current_X)
                    future_predictions.append(predicted_price[0][0])
                    
                    # Update the input window for the next prediction
                    current_X = np.append(current_X[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

                # Inverse transform the predictions
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

                # Generate dates starting from day after tomorrow
                start_date = df.index[-1] + pd.Timedelta(days=2)
                future_dates = pd.date_range(start=start_date, periods=days_to_predict)
                prediction_df = pd.DataFrame(future_predictions, columns=['Predicted Price'], index=future_dates)
                
                # Combine the last 5 days of historical data and predicted data for plotting
                historical_data = df['Close'].iloc[-days_to_predict:]  # Get the last 5 days of historical data
                combined_df = pd.concat([historical_data, prediction_df])
                
                # Plot using Streamlit's line_chart
                st.subheader(f"📈 Historical and Predicted Stock Price Trend for Next {days_to_predict} Days")
                st.line_chart(combined_df)

                # Display the predicted prices in a table
                st.subheader(f"🔮 Predicted Stock Prices for Next {days_to_predict} Days")
                st.dataframe(prediction_df.style.format("{:.2f}"))

                # Calculate trading signals using the first prediction
                latest_close = df['Close'].iloc[-1]
                predicted_price = future_predictions[0]
                price_change = ((predicted_price - latest_close) / latest_close) * 100

                st.subheader("⚠️ Trading Signal")
                if price_change > 1:
                    st.success(f"Buy Signal: Price expected to rise by {price_change:.2f}%")
                elif price_change < -1:
                    st.error(f"Sell Signal: Price expected to drop by {price_change:.2f}%")
                else:
                    st.warning(f"Hold Signal: Price expected to change by {price_change:.2f}%")
                
                st.subheader("📢 Disclaimer")
                st.warning("The stock price predictions and data presented on this website are generated using artificial intelligence and machine learning models, including LSTM (Long Short-Term Memory) networks. While we strive for accuracy and continuous improvement, these predictions are for informational and educational purposes only and do not constitute financial advice. \n\nWe do not guarantee the accuracy, completeness, or reliability of the predictions or any related content. Stock prices are inherently volatile and influenced by numerous unpredictable factors. Past performance is not indicative of future results. \n\nBrokers, investors, and all users are strongly advised to conduct their own independent research and consult with licensed financial advisors before making any investment decisions. Any reliance on the information provided here is done entirely at your own risk. \n\nBy using this site, you acknowledge and agree that the owners, developers, and contributors of this platform are not responsible for any losses or damages arising from your use of the information provided.")                
                
            else:
                st.error("Model failed to load. Prediction not available.")
                
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        
# Add developer details and copyright information
st.markdown("---")
st.markdown("🏆 Developed by **Shreyas Meru**")
st.markdown("🌐 [Visit My Website](https://shreyas-meru.web.app/)")
st.markdown("© 2025 Shreyas Meru. All rights reserved.")