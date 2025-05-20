# 📈 Stock Price Trend Prediction with LSTM

This project implements a stock price trend prediction model using LSTM (Long Short-Term Memory) neural networks. The model is integrated into a **Streamlit web application**, allowing users to view real-time historical data, technical indicators, and make short-term price predictions for any publicly traded stock using its ticker symbol.

> **Website**: [https://stocktrendai.streamlit.app/](https://stocktrendai.streamlit.app/)

---

## 🚀 Features

- 🔍 Real-time stock data from Yahoo Finance
- 📊 Visualization of Open, High, Low, Close, Volume
- 📉 50-Day Moving Average, RSI, and MACD indicators
- 🤖 Predict stock prices for 1–30 future days using a trained LSTM model
- ⚠️ Generates Buy / Hold / Sell signals
- 💻 Clean and responsive UI built with Streamlit

---

## 🧠 Model Architecture

The LSTM model is trained on the past 60 days of closing prices to predict the next day's price. For multiple future days, the model feeds its own predictions into the next iteration.

- **Input**: Normalized closing prices for the past 60 days
- **Model**: LSTM → Dense
- **Loss Function**: Mean Squared Error (MSE)
- **Scaler**: MinMaxScaler (from scikit-learn)

Model is stored in:  
`model_weights.h5`

---

## 📦 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, TensorFlow (Keras), scikit-learn
- **Data**: yFinance API
- **Visualization**: Streamlit Charts
- **Indicators**: `ta` library (RSI, MACD)

---

## 📸 Screenshots

![Main Dashboard](https://github.com/user-attachments/assets/cc3f8426-2546-4e3d-b9fb-d3b78e29d1cb)  
*Main dashboard with input fields and prediction controls*

![Stock Prediction](https://github.com/user-attachments/assets/efe34b4e-ed5e-4f90-b956-1e6302fb79af)
*Line chart of past 4 days and predicted price trend*

---

## ⚙️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/Shreyas-Meru/Stock-Price-Trend-Prediction-with-LSTM.git
cd Stock-Price-Trend-Prediction-with-LSTM
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run streamlit_dashboard.py
```

---

## 📁 Project Structure

```
├── model_weights.h5             # Pre-trained LSTM model
├── requirements.txt             # Python dependencies
├── streamlit_dashboard.py       # Main Streamlit application
├── stock_lstm.ipynb             # Jupyter Notebook for model training
└── README.md
```

---

## ⚠️ Disclaimer

> The stock price predictions and data presented on this website are generated using artificial intelligence and machine learning models, including LSTM (Long Short-Term Memory) networks. While we strive for accuracy and continuous improvement, these predictions are for informational and educational purposes only and do not constitute financial advice.

> We do not guarantee the accuracy, completeness, or reliability of the predictions or any related content. Stock prices are inherently volatile and influenced by numerous unpredictable factors. Past performance is not indicative of future results.

> Brokers, investors, and all users are strongly advised to conduct their own independent research and consult with licensed financial advisors before making any investment decisions. Any reliance on the information provided here is done entirely at your own risk.

> By using this site, you acknowledge and agree that the owners, developers, and contributors of this platform are not responsible for any losses or damages arising from your use of the information provided.

---

## 👨‍💻 Author

**Shreyas Meru**
🔗 [Portfolio](https://shreyas-meru.web.app/)
📫 [LinkedIn](https://www.linkedin.com/in/shreyasmeru/)

---

## ⭐️ Give it a Star!

If you like this project, feel free to ⭐ the repo and share it with others!
