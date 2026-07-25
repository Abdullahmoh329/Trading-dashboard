import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# 1. تهيئة الصفحة والتصميم Visual Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Quant Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px; text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; }
    .metric-val { font-size: 1.3rem; font-weight: bold; color: #f0f6fc; }
    .call-banner {
        background: rgba(46, 160, 67, 0.15); border: 2px solid #2ea043;
        border-radius: 10px; padding: 16px; margin-bottom: 20px;
    }
    .put-banner {
        background: rgba(218, 54, 51, 0.15); border: 2px solid #da3633;
        border-radius: 10px; padding: 16px; margin-bottom: 20px;
    }
    .wait-banner {
        background: rgba(139, 148, 158, 0.15); border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. شريط الإعدادات Sidebar
# -------------------------------------------------------------
st.sidebar.title("⚡ Quant Terminal")
st.sidebar.caption("محرك تحليل الاختراقات وإشارات السكالبينغ")

ticker_symbol = st.sidebar.text_input("رمز السهم (مثل NVDA, TSLA, AMD):", value="NVDA").upper().strip()
timeframe = st.sidebar.selectbox("الفريم الزمني:", ["5m", "15m"], index=1)
vol_multiplier = st.sidebar.slider("مضاعف حجم التداول المطلوبة (Vol Spike):", 1.1, 2.5, 1.4, 0.1)
atr_sl_mult = st.sidebar.slider("مضاعف وقف الخسارة (ATR SL):", 0.8, 2.5, 1.2, 0.1)

# -------------------------------------------------------------
# 3. محرك جلب البيانات وتحليلها Data Processing Engine
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_and_analyze_data(symbol, interval, vol_mult):
    try:
        # جلب أحدث بيانات intraday
        df = yf.download(symbol, period="5d", interval=interval, progress=False)
        
        if df.empty:
            return None, "لا توجد بيانات متاحة لهذا الرمز."

        # تنظيف أعمدة MultiIndex من yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        if len(df) < 30:
            return None, "عدد الشموع المجلوبة غير كافٍ للتحليل."

        # أ) حساب الـ VWAP اللحظي المترسب يومياً
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['TP'] * df['Volume']
        df['Date'] = df.index.date
        df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
        df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
        df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']

        # ب) المتوسطات المتحركة EMA
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # ج) حجم التداول والانفجار السعري
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Vol_Spike'] = df['Volume'] > (df['Vol_MA'] * vol_mult)
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']

        # د) مؤشر ATR لوضع الأهداف ووقف الخسارة
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # هـ) الدعوم والمقاومات
        df['Resistance'] = df['High'].rolling(20).max().shift(1)
        df['Support'] = df['Low'].rolling(20).min().shift(1)

        return df, None
    except Exception as e:
        return None, f"حدث خطأ أثناء جلب البيانات: {str(e)}"

# -------------------------------------------------------------
# 4. تنفيذ التحليل وعرض النشرة
# -------------------------------------------------------------
df, err_msg = fetch_and_analyze_data(ticker_symbol, timeframe, vol_multiplier)

if err_msg:
    st.error(f"⚠️ {err_msg}")
else:
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    curr_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])
    atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else curr_price * 0.01

    # الخوارزمية الكمية لحساب النقاط (0 - 100)
    score_call = 0
    score_put = 0
    reasons = []

    # 1. معيار الـ VWAP
    if curr_price > vwap_val:
        score_call += 25
        reasons.append("السعر أعلى من الـ VWAP اللحظي (سيطرة المؤسسات والقوة الشرائية).")
    else:
        score_put += 25
        reasons.append("السعر أسفل الـ VWAP اللحظي (ضغط بيعي مؤسسي).")

    # 2. معيار اتجاه المتوسطات EMA
    if latest['EMA_9'] > latest['EMA_21'] > latest['EMA_50']:
        score_call += 25
        reasons.append("ترتيب متناسق صاعد للمتوسطات السريعة (EMA 9 > 21 > 50).")
    elif latest['EMA_9'] < latest['EMA_21'] < latest['EMA_50']:
        score_put += 25
        reasons.append("ترتيب متناسق هابط للمتوسطات السريعة (EMA 9 < 21 < 50).")
    else:
        reasons.append("المتوسطات السعرية في حالة تداخل وتذبذب جانبي.")

    # 3. معيار الفوليوم والسيولة
    if latest['Vol_Spike']:
        if curr_price > latest['Open']:
            score_call += 25
            reasons.append(f"تدفق سيولة شرائية عالية ({latest['Vol_Ratio']:.1f}x ضعف المتوسط).")
        else:
            score_put += 25
            reasons.append(f"تدفق سيولة بيعية عالية ({latest['Vol_Ratio']:.1f}x ضعف المتوسط).")

    # 4. معيار إعادة الاختبار (Retest)
    if prev['Close'] > prev['Resistance'] and latest['Low'] <= latest['Resistance'] * 1.001 and curr_price >= latest['Resistance']:
        score_call += 25
        reasons.append("🎯 توثيق إعادة اختبار (Retest) ناجح للمقاومة المكسورة وتحولها إلى دعم.")
    elif prev['Close'] < prev['Support'] and latest['High'] >= latest['Support'] * 0.999 and curr_price <= latest['Support']:
        score_put += 25
        reasons.append("🎯 توثيق إعادة اختبار (Retest) ناجح للدعم المكسور وتحوله إلى مقاومة.")

    # تحديد التوصية المستخرجة
    if score_call >= 60:
        action = "STRONG CALL (شراء عقد صاعد)"
        banner_class = "call-banner"
        action_color = "#2ea043"
        sl = round(curr_price - (atr_sl_mult * atr_val), 2)
        tp1 = round(curr_price + (1.5 * atr_val), 2)
        tp2 = round(curr_price + (2.5 * atr_val), 2)
        confidence = score_call
    elif score_put >= 60:
        action = "STRONG PUT (شراء عقد هابط)"
        banner_class = "put-banner"
        action_color = "#da3633"
        sl = round(curr_price + (atr_sl_mult * atr_val), 2)
        tp1 = round(curr_price - (1.5 * atr_val), 2)
        tp2 = round(curr_price - (2.5 * atr_val), 2)
        confidence = score_put
    else:
        action = "WAIT / NEUTRAL (انتظار - عدم دخول)"
        banner_class = "wait-banner"
        action_color = "#8b949e"
        sl = round(curr_price * 0.99, 2)
        tp1 = round(curr_price * 1.01, 2)
        tp2 = round(curr_price * 1.02, 2)
        confidence = max(score_call, score_put)

    rr_ratio = round(abs(tp1 - curr_price) / abs(curr_price - sl), 2) if abs(curr_price - sl) > 0 else 1.5

    # -------------------------------------------------------------
    # 5. عرض الواجهة الرئيسية Dashboard UI
    # -------------------------------------------------------------
    st.title(f"📈 غرفة تداول: {ticker_symbol} ({timeframe})")
    
    # كارت التوصية المباشرة
    st.markdown(f"""
    <div class="{banner_class}">
        <h2 style="margin:0; color:{action_color};">{action}</h2>
        <p style="margin:6px 0 0 0; font-size:1rem;">
            السعر الحالي: <b>${curr_price:.2f}</b> | خط الـ VWAP: <b>${vwap_val:.2f}</b> | درجة الثقة: <b>{confidence}/100</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # صف الأرقام والأهداف
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">سعر الدخول المقترح</div><div class="metric-val" style="color:#58a6ff;">${curr_price:.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">وقف الخسارة (SL)</div><div class="metric-val" style="color:#f85149;">${sl:.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">الهدف الأول (TP1)</div><div class="metric-val" style="color:#3fb950;">${tp1:.2f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">الهدف الثاني (TP2)</div><div class="metric-val" style="color:#2ea043;">${tp2:.2f}</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-label">نسبة العائد/المخاطرة</div><div class="metric-val">1:{rr_ratio}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # التبويبات التفاعلية
    tab_chart, tab_reasons = st.tabs(["📊 الرسم البياني التفاعلي", "💡 مبررات التحليل والسيولة"])

    with tab_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

        # الشموع
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="السعر", increasing_line_color='#2ea043', decreasing_line_color='#da3633'
        ), row=1, col=1)

        # الـ VWAP
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#f1e05a', width=1.5), name="VWAP"), row=1, col=1)

        # المتوسطات
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='#58a6ff', width=1), name="EMA 9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='#bc8cff', width=1), name="EMA 21"), row=1, col=1)

        # أحجام التداول
        colors = np.where(df['Vol_Spike'], '#00f5ff', np.where(df['Close'] >= df['Open'], '#2ea043', '#da3633'))
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", height=580,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_reasons:
        st.subheader("💡 تفاصيل التحليل الكمي ومبررات الإشارة:")
        for r in reasons:
            st.markdown(f"- {r}")
