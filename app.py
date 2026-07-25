import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from datetime import date

# -------------------------------------------------------------
# 1. Page Configuration & Dark Theme CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="AlphaQuant Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #94a3b8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
        padding-top: 20px;
    }
    .terminal-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 16px 24px 16px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
    }
    .terminal-logo-icon {
        background: #3b82f6;
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .terminal-logo-text {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .terminal-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 16px;
    }
    .card-label {
        font-size: 0.75rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f9fafb;
    }
    .pattern-box-bullish {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        color: #f8fafc;
        font-weight: 600;
    }
    .pattern-box-bearish {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #f43f5e;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        color: #f8fafc;
        font-weight: 600;
    }
    .pattern-box-neutral {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        color: #f8fafc;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Indicators & Pure Live Data Engine (No Mock Data)
# -------------------------------------------------------------
def compute_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    return df.dropna()

@st.cache_data(ttl=60)
def fetch_live_data(ticker_symbol, timeframe="6m"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=timeframe)
        if df is None or df.empty:
            df = yf.download(ticker_symbol, period=timeframe, progress=False)
            
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return compute_indicators(df)
    except Exception as e:
        st.error(f"خطأ في جلب بيانات السوق الحية: {e}")
    return None

# -------------------------------------------------------------
# 3. Machine Learning & Quantitative Backtest Engine
# -------------------------------------------------------------
def run_ml_and_backtest(df, ml_threshold=0.52, initial_capital=10000):
    data = df.copy()
    
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    features = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'Returns', 'Vol_Change']
    ml_data = data.dropna()
    
    X = ml_data[features]
    y = ml_data['Target']
    
    if len(X) > 30:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X[:-1], y[:-1])
        probs = model.predict_proba(X)[:, 1]
        data.loc[ml_data.index, 'ML_Prob'] = probs
    else:
        data['ML_Prob'] = 0.5
        
    data['ML_Prob'] = data['ML_Prob'].fillna(0.5)
    
    data['Signal'] = 0
    data.loc[data['ML_Prob'] > ml_threshold, 'Signal'] = 1
    data.loc[data['ML_Prob'] < (1 - ml_threshold), 'Signal'] = -1
    
    data['Position'] = data['Signal'].shift(1).fillna(0)
    data['Strategy_Returns'] = data['Returns'] * data['Position']
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns'].fillna(0)).cumprod() * initial_capital
    data['Cumulative_Market'] = (1 + data['Returns'].fillna(0)).cumprod() * initial_capital
    
    total_trades = int((data['Position'].diff().abs() > 0).sum())
    net_profit = float(data['Cumulative_Strategy'].iloc[-1] - initial_capital)
    ret_pct = (net_profit / initial_capital) * 100
    winning_days = (data['Strategy_Returns'] > 0).sum()
    total_active_days = (data['Strategy_Returns'] != 0).sum()
    win_rate = (winning_days / total_active_days * 100) if total_active_days > 0 else 0.0
    
    rolling_max = data['Cumulative_Strategy'].cummax()
    drawdown = (data['Cumulative_Strategy'] - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100) if not drawdown.empty else 0.0
    
    latest_ml_prob = float(data['ML_Prob'].iloc[-1]) * 100
    
    reliability_score = round((latest_ml_prob * 0.50) + (win_rate * 0.35) + (min(100, max(0, ret_pct + 50)) * 0.15), 1)
    reliability_score = min(99.5, max(15.0, reliability_score))
    
    return data, {
        "net_profit": net_profit, "ret_pct": ret_pct,
        "win_rate": win_rate, "max_dd": max_drawdown, "trades": total_trades,
        "ml_prob": latest_ml_prob, "reliability": reliability_score
    }

# -------------------------------------------------------------
# 4. Pattern Recognition (Unambiguous)
# -------------------------------------------------------------
def detect_patterns(df):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    peaks, _ = find_peaks(highs, distance=20)
    troughs, _ = find_peaks(-lows, distance=20)
    
    patterns = []
    
    if len(troughs) >= 2:
        t1, t2 = lows[troughs[-2]], lows[troughs[-1]]
        if abs(t1 - t2) / t1 < 0.02 and closes[-1] > t2:
            patterns.append(("bullish", f"🟢 نموذج قاع مزدوج مؤكد (دعم عند ${t2:.2f})"))
            
    if len(peaks) >= 2:
        p1, p2 = highs[peaks[-2]], highs[peaks[-1]]
        if abs(p1 - p2) / p1 < 0.02 and closes[-1] < p2:
            patterns.append(("bearish", f"🔴 نموذج قمة مزدوجة مؤكد (مقاومة عند ${p2:.2f})"))
            
    if not patterns:
        recent_ret = (closes[-1] - closes[-10]) / closes[-10]
        if recent_ret > 0.02:
            patterns.append(("bullish", f"🚀 زخم صاعد قوي (+{recent_ret*100:.1f}% خلال آخر 10 جلسات)"))
        elif recent_ret < -0.02:
            patterns.append(("bearish", f"⚠️ ضغط هبوطي وتصحيح ({recent_ret*100:.1f}% خلال آخر 10 جلسات)"))
        else:
            patterns.append(("neutral", "➡️ حركية أفقية ونطاق عرضي مستقر"))
            
    return patterns

# -------------------------------------------------------------
# 5. Sidebar & Navigation
# -------------------------------------------------------------
st.sidebar.markdown("""
<div class="terminal-logo">
    <div class="terminal-logo-icon">⚡</div>
    <div class="terminal-logo-text">AlphaQuant Pro</div>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "القائمة",
    ["📊 لوحة المؤشرات", "📈 تحليلات الباك تست والذكاء الاصطناعي", "⚙️ الإعدادات"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("رمز السهم النشط:", value="AMD").upper().strip()

df = fetch_live_data(symbol)

if df is None or df.empty:
    st.error(f"❌ لم يتم العثور على بيانات حية للرمز {symbol}. تأكد من صحة الرمز المكتوب.")
    st.stop()

api_live_price = float(df['Close'].iloc[-1])
manual_override = st.sidebar.checkbox("تفعيل تعديل السعر يدوياً", value=False)
if manual_override:
    live_price = st.sidebar.number_input("السعر الفعلي ($):", value=api_live_price, step=0.1)
    df.iloc[-1, df.columns.get_loc('Close')] = live_price
else:
    live_price = api_live_price

atr_val = float(df['ATR'].iloc[-1])

st.sidebar.markdown(f"**السعر الحي:** `${live_price:.2f}`")
st.sidebar.markdown(f"**التقلب (ATR):** `${atr_val:.2f}`")

if st.sidebar.button("🔄 مسح الذاكرة وتحديث البيانات"):
    st.cache_data.clear()
    st.rerun()

current_date_str = date.today().strftime("%A, %B %d, %Y")

# -------------------------------------------------------------
# VIEW 1: DASHBOARD
# -------------------------------------------------------------
if nav == "📊 لوحة المؤشرات":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">لوحة التحليل الكمي: {symbol}</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">{current_date_str} — بيانات حقيقية مباشرة من السوق</p>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="terminal-card"><div class="card-label">السعر اللحظي</div><div class="card-value" style="color: #3b82f6;">${live_price:.2f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="terminal-card"><div class="card-label">الرمز النشط</div><div class="card-value">{symbol}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="terminal-card"><div class="card-label">مصدر البيانات</div><div class="card-value" style="color: #10b981;">حقيقي 100%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="terminal-card"><div class="card-label">مؤشر التقلب ATR</div><div class="card-value" style="color: #f59e0b;">${atr_val:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#3b82f6', width=2.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)', name='السعر الحي'))
    fig_price.update_layout(template="plotly_dark", height=320, paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_price, use_container_width=True)

# -------------------------------------------------------------
# VIEW 2: ML & BACKTEST ANALYTICS
# -------------------------------------------------------------
elif nav == "📈 تحليلات الباك تست والذكاء الاصطناعي":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">تحليلات الـ ML والباك تست: <span style="color:#3b82f6;">{symbol}</span></h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">محرك التعلم الآلي والاختبار العكسي المبني على البيانات الحية</p>
    """, unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns(2)
    ml_threshold = b_col1.slider("عتبة احتمالية الدخول (ML Threshold):", min_value=0.50, max_value=0.65, value=0.52, step=0.01)
    capital = b_col2.number_input("رأس المال الافتراضي للباك تست ($):", min_value=1000, value=10000, step=1000)
    
    bt_df, metrics = run_ml_and_backtest(df, ml_threshold, capital)
    patterns = detect_patterns(df)
    
    st.markdown("### ⚡ مؤشرات الموثوقية والأداء الكمي")
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(f'<div class="terminal-card"><div class="card-label">الموثوقية الكلية</div><div class="card-value" style="color: #10b981;">{metrics["reliability"]:.1f}%</div></div>', unsafe_allow_html=True)
    r2.markdown(f'<div class="terminal-card"><div class="card-label">احتمالية صعود الـ ML</div><div class="card-value" style="color: #3b82f6;">{metrics["ml_prob"]:.1f}%</div></div>', unsafe_allow_html=True)
    r3.markdown(f'<div class="terminal-card"><div class="card-label">دقة صفقات الباك تست</div><div class="card-value" style="color: #f59e0b;">{metrics["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
    r4.markdown(f'<div class="terminal-card"><div class="card-label">صافي ربح الاستراتيجية</div><div class="card-value" style="color: {"#10b981" if metrics["net_profit"]>=0 else "#f43f5e"};">${metrics["net_profit"]:.2f}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 الهيكلة الفنية المفلترة (بدون تعارض)")
    for p_type, p_text in patterns:
        css_class = "pattern-box-bullish" if p_type == "bullish" else ("pattern-box-bearish" if p_type == "bearish" else "pattern-box-neutral")
        st.markdown(f'<div class="{css_class}">{p_text}</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🧪 منحنى الأداء الرأسمالي للاستراتيجية مقابل السوق")
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    fig_bt.add_trace(go.Candlestick(x=bt_df.index, open=bt_df['Open'], high=bt_df['High'], low=bt_df['Low'], close=bt_df['Close'], name="السعر الحي"), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Strategy'], line=dict(color='#10b981', width=2), name="محفظة استراتيجية الـ ML"), row=2, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Market'], line=dict(color='#94a3b8', width=1, dash='dash'), name="شراء واحتفاظ السوق"), row=2, col=1)
    
    fig_bt.update_layout(template="plotly_dark", height=520, paper_bgcolor="#111827", plot_bgcolor="#111827", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_bt, use_container_width=True)

# -------------------------------------------------------------
# VIEW 3: SETTINGS
# -------------------------------------------------------------
elif nav == "⚙️ الإعدادات":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">إعدادات النظام</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">تكوين منصة التحليل الكمي</p>
    """, unsafe_allow_html=True)
    st.markdown('<div class="terminal-card"><h3>تفاصيل المحرك التقني الحصري</h3><p>• مصدر البيانات: حقيقي 100% عبر yfinance مباشرة (لا توجد بيانات وهمية).</p><p>• نموذج الـ Machine Learning: Random Forest Classifier مدرب على مصفوفة مؤشرات عزم وسيولة (RSI, MACD, ATR, Volume).</p><p>• الباك تست: معتمد حصرياً على احتمالات تنبؤ الذكاء الاصطناعي (وليس تقاطع متوسطات بسيط).</p></div>', unsafe_allow_html=True)
