import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Trading Dashboard & ML", layout="wide")

st.title("📈 لوحة تحكم التداول وتوقعات الذكاء الاصطناعي")

# القائمة الجانبية واختيار الأسهم
st.sidebar.header("إعدادات السهم")
stocks = {
    "أبل (AAPL)": "AAPL",
    "تسلا (TSLA)": "TSLA",
    "إنفيديا (NVDA)": "NVDA",
    "إي إم دي (AMD)": "AMD",
    "سابك (2010.SR)": "2010.SR",
    "مسك (2370.SR)": "2370.SR",
    "أيان (2140.SR)": "2140.SR"
}

selected_stock_name = st.sidebar.selectbox("اختر السهم للمتابعة:", list(stocks.keys()))
ticker = stocks[selected_stock_name]

# جلب البيانات
@st.cache_data(ttl=300)
def load_data(symbol):
    df = yf.Ticker(symbol).history(period="1y")
    return df

df = load_data(ticker)

if df.empty:
    st.error("تعذر جلب البيانات للسهم المختار، يرجى المحاولة لاحقاً.")
else:
    # 1. حساب المؤشرات الفنية
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 2. تدريب نموذج تعلم الآلة (ML Prediction)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    features = ['RSI', 'SMA_20', 'SMA_50', 'MACD']
    
    clean_df = df.dropna()
    X = clean_df[features]
    y = clean_df['Target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-1], y[:-1]) # تدريب النموذج
    
    latest_features = X.iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    prob = model.predict_proba(latest_features)[0]

    # 3. عرض النتائج والمؤشرات الحية
    col1, col2, col3, col4 = st.columns(4)
    last_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100
    
    col1.metric("السعر الحالي", f"${last_price:.2f}" if "SR" not in ticker else f"{last_price:.2f} SAR", f"{pct_change:.2f}%")
    col2.metric("مؤشر القوة النسبية (RSI)", f"{df['RSI'].iloc[-1]:.1f}")
    col3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    
    # إشارة الذكاء الاصطناعي
    if prediction == 1:
        col4.success(f"🤖 إشارة ML: شراء/صعود (ثقة: {prob[1]*100:.0f}%)")
    else:
        col4.error(f"🤖 إشارة ML: بيع/هبوط (ثقة: {prob[0]*100:.0f}%)")

    # 4. رسم الشارت التفاعلي (Plotly Candlestick)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="السعر"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='blue', width=1), name="SMA 50"))
    
    fig.update_layout(title=f"شارت {selected_stock_name}", yaxis_title="السعر", template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)
