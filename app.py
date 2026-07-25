import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# 1. تهيئة الواجهة Visual Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Institutional AI & Options Quant Terminal", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; text-align: center; }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; }
    .metric-val { font-size: 1.2rem; font-weight: bold; color: #f0f6fc; }
    .call-banner { background: rgba(46, 160, 67, 0.15); border: 2px solid #2ea043; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
    .put-banner { background: rgba(218, 54, 51, 0.15); border: 2px solid #da3633; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
    .wait-banner { background: rgba(139, 148, 158, 0.15); border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. القائمة الجانبية وحاسبة الأوبشنز
# -------------------------------------------------------------
st.sidebar.title("⚡ Options Quant Terminal")
st.sidebar.caption("محرك تداول كمي بالذكاء الاصطناعي وحاسبة أرباح الأوبشنز")

ticker_symbol = st.sidebar.text_input("رمز السهم:", value="NVDA").upper().strip()
timeframe = st.sidebar.selectbox("الفريم الزمني:", ["5m", "15m", "1h"], index=1)
min_ml_prob = st.sidebar.slider("حد ثقة الذكاء الاصطناعي (ML Prob %):", 50, 85, 65, 5) / 100

st.sidebar.markdown("---")
st.sidebar.header("🎯 حاسبة عقود الأوبشنز (Options Calculator)")
option_price = st.sidebar.number_input("سعر عقد الأوبشن الحالي ($):", value=2.50, step=0.10)
option_delta = st.sidebar.slider("معامل الدلتا (Delta Δ):", 0.10, 0.95, 0.50, 0.05)

# -------------------------------------------------------------
# 3. حساب المؤشرات وتجهيز بيانات الذكاء الاصطناعي
# -------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=60)
def load_data(symbol, interval):
    df = yf.download(symbol, period="1mo" if interval in ["5m", "15m"] else "6mo", interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['VWAP'] = df.groupby('Date')['VP'].cumsum() / df.groupby('Date')['Volume'].cumsum()

    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # ATR لحساب الأهداف
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # خصائص الذكاء الاصطناعي
    df['Feature_Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['Feature_EMA_Diff'] = (df['EMA_9'] - df['EMA_21']) / df['EMA_21']
    df['Feature_RSI'] = df['RSI']
    df['Feature_Vol_Ratio'] = df['Vol_Ratio']

    df['Future_Return'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = np.where(df['Future_Return'] > 0.008, 1, 0)

    return df.dropna()

df = load_data(ticker_symbol, timeframe)

# -------------------------------------------------------------
# 4. تدريب النموذج والتحليل
# -------------------------------------------------------------
if not df.empty and len(df) >= 100:
    feature_cols = ['Feature_Dist_VWAP', 'Feature_EMA_Diff', 'Feature_RSI', 'Feature_Vol_Ratio']
    X = df[feature_cols]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    model_acc = rf_model.score(X_test, y_test)

    latest = df.iloc[-1]
    curr_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])
    atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else curr_price * 0.01

    current_features = np.array([[latest['Feature_Dist_VWAP'], latest['Feature_EMA_Diff'], latest['Feature_RSI'], latest['Feature_Vol_Ratio']]])
    win_prob_call = rf_model.predict_proba(current_features)[0][1]
    win_prob_put = 1.0 - win_prob_call

    technical_call = (curr_price > vwap_val) and (latest['EMA_9'] > latest['EMA_21'])
    technical_put = (curr_price < vwap_val) and (latest['EMA_9'] < latest['EMA_21'])

    # -------------------------------------------------------------
    # 5. حسابات الأوبشنز وحساب المتوقع بناءً على Delta
    # -------------------------------------------------------------
    # أهداف تحرك السهم
    stock_target_call = curr_price + (1.5 * atr_val)
    stock_sl_call = curr_price - (1.0 * atr_val)

    stock_target_put = curr_price - (1.5 * atr_val)
    stock_sl_put = curr_price + (1.0 * atr_val)

    # حساب أرباح/خسائر العقد بالـ Delta
    if technical_call or win_prob_call >= win_prob_put:
        stock_change_tp = stock_target_call - curr_price
        stock_change_sl = curr_price - stock_sl_call
    else:
        stock_change_tp = curr_price - stock_target_put
        stock_change_sl = stock_sl_put - curr_price

    option_gain_dollar = stock_change_tp * option_delta
    option_loss_dollar = stock_change_sl * option_delta

    option_target_price = option_price + option_gain_dollar
    option_sl_price = max(0.01, option_price - option_loss_dollar)

    option_tp_pct = (option_gain_dollar / option_price) * 100 if option_price > 0 else 0
    option_sl_pct = (option_loss_dollar / option_price) * 100 if option_price > 0 else 0

    # تحديد القرار
    if technical_call and win_prob_call >= min_ml_prob:
        action = f"STRONG CALL 📈 (ثقة الذكاء الاصطناعي: {win_prob_call*100:.1f}%)"
        banner = "call-banner"
        color = "#2ea043"
    elif technical_put and win_prob_put >= min_ml_prob:
        action = f"STRONG PUT 📉 (ثقة الذكاء الاصطناعي: {win_prob_put*100:.1f}%)"
        banner = "put-banner"
        color = "#da3633"
    else:
        action = f"WAIT / NEUTRAL ⏳ (احتمالية النجاح {max(win_prob_call, win_prob_put)*100:.1f}% أقل من الحد المطلوبة)"
        banner = "wait-banner"
        color = "#8b949e"

    # -------------------------------------------------------------
    # 6. العرض الرئيسي Dashboard
    # -------------------------------------------------------------
    st.title(f"📊 غرفة تداول الأوبشنز الكمية: {ticker_symbol}")

    st.markdown(f"""
    <div class="{banner}">
        <h2 style="margin:0; color:{color};">{action}</h2>
        <p style="margin:5px 0 0 0;">سعر السهم الحالي: <b>${curr_price:.2f}</b> | VWAP: <b>${vwap_val:.2f}</b> | دقة النموذج التاريخية: <b>{model_acc*100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # عرض كروت محاكي الأوبشنز
    st.subheader("🎯 محاكي أرباح وخسائر عقد الأوبشن (Option PnL Estimator)")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">سعر شراء العقد الحالي</div><div class="metric-val">${option_price:.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">هدف بيع العقد (TP)</div><div class="metric-val" style="color:#2ea043;">${option_target_price:.2f} (+{option_tp_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">وقف خسارة العقد (SL)</div><div class="metric-val" style="color:#da3633;">${option_sl_price:.2f} (-{option_sl_pct:.1f}%)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">ربح العقد المتوقع ($)</div><div class="metric-val" style="color:#58a6ff;">+${option_gain_dollar*100:.0f} / للعقد</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # الرسم البياني
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#f1e05a', width=1.5), name="VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#58a6ff', width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("جاري جلب البيانات أو أن عدد الشموع المجلوبة غير كافٍ للتحليل.")
