import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# 1. تهيئة الصفحة والتنسيق Visual Styling (Dark Quant Theme)
# -------------------------------------------------------------
st.set_page_config(page_title="Quant Scalping Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* خلفية التطبيق العامة */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* تصميم بطاقات الإحصائيات Custom Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #f0f6fc;
    }
    /* بطاقة التوصية الحية Signal Alert Box */
    .signal-card-buy {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.2), rgba(13, 17, 23, 0.9));
        border: 2px solid #2ea043;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .signal-card-neutral {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. القائمة الجانبية والإعدادات Sidebar Controls
# -------------------------------------------------------------
st.sidebar.title("⚡ Quant Terminal")
st.sidebar.caption("محرك السكالبينغ الموحد للأسهم والعقود")

st.sidebar.markdown("### 1️⃣ بيانات السوق")
ticker_input = st.sidebar.text_input("رمز السهم:", value="NVDA").upper().strip()
interval_selected = st.sidebar.selectbox("الفريم الزمني:", ["5m", "15m", "1d"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ خوارزمية التداول")
strategy_choice = st.sidebar.selectbox(
    "الاستراتيجية الرئيسية:",
    [
        "اختراق الدعم/المقاومة + إعادة الاختبار (S/R Retest)",
        "اختراق خط الترند الديناميكي (Trendline Breakout)",
        "تقاطع المتوسطات السريعة (EMA Scalper 9/21)"
    ]
)

# فلتر المؤسسات الإضافي
use_vwap_filter = st.sidebar.checkbox("تفعيل فلتر السيولة والـ VWAP", value=True, help="يشترط أن يكون السعر أعلى من الـ VWAP والفوليوم أعلى من المتوسط لتوثيق الإشارة.")
vol_multiplier = st.sidebar.slider("مضاعف الفوليوم المطلوب (X):", 1.1, 2.5, 1.3, 0.1) if use_vwap_filter else 1.0

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ إدارة المخاطر")
stop_loss_pct = st.sidebar.slider("وقف الخسارة (Stop Loss %):", 0.3, 3.0, 0.8, 0.1) / 100
take_profit_pct = st.sidebar.slider("هدف الربح (Take Profit %):", 0.6, 6.0, 1.6, 0.1) / 100

# -------------------------------------------------------------
# 3. جلب البيانات وحساب المؤشرات Data Engine
# -------------------------------------------------------------
@st.cache_data(ttl=180)
def load_market_data(symbol, interval):
    period_map = {"5m": "1mo", "15m": "1mo", "1d": "1y"}
    try:
        df = yf.download(symbol, period=period_map[interval], interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        return pd.DataFrame()

df = load_market_data(ticker_input, interval_selected)

if df.empty or len(df) < 50:
    st.error(f"⚠️ تعذر جلب بيانات السهم '{ticker_input}'. تأكد من الرمز والفريم الزمني.")
else:
    # --- حسابات المؤشرات الفنية الأساسية ---
    # 1. VWAP اللحظي
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['Date_Only'] = df.index.date
    df['Cum_VP'] = df.groupby('Date_Only')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date_Only')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']

    # 2. الفوليوم والمتوسط
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Vol_MA'] * vol_multiplier)

    # 3. مستويات الدعم والمقاومة
    lookback = 20
    df['Resistance'] = df['High'].rolling(window=lookback).max().shift(1)
    df['Support'] = df['Low'].rolling(window=lookback).min().shift(1)

    # 4. المتوسطات المتحركة
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # -------------------------------------------------------------
    # 4. منطق إشارات الاستراتيجيات (Signal Rules)
    # -------------------------------------------------------------
    df['Signal'] = 0

    if strategy_choice == "اختراق الدعم/المقاومة + إعادة الاختبار (S/R Retest)":
        # شرط الاختراق + إعادة الاختبار:
        # الشمعه السابقة كسر المقاومة -> الشمعة الحالية هبطت لتختبر المقاومة وأغلقت صاعدة
        breakout = df['Close'].shift(1) > df['Resistance'].shift(1)
        retest = (df['Low'] <= df['Resistance'] * 1.0015) & (df['Close'] >= df['Resistance'])
        bullish_candle = df['Close'] > df['Open']
        base_signal = breakout & retest & bullish_candle

    elif strategy_choice == "اختراق خط الترند الديناميكي (Trendline Breakout)":
        df['Rolling_Max'] = df['High'].rolling(window=12).max()
        df['Trendline'] = df['Rolling_Max'].shift(1)
        base_signal = (df['Close'] > df['Trendline']) & (df['Close'].shift(1) <= df['Trendline'].shift(1))

    elif strategy_choice == "تقاطع المتوسطات السريعة (EMA Scalper 9/21)":
        base_signal = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1))

    # تطبيق الفلتر المؤسسي (VWAP + Volume) إذا تم تفعيله
    if use_vwap_filter:
        institutional_filter = (df['Close'] > df['VWAP']) & df['Volume_Spike']
        df.loc[base_signal & institutional_filter, 'Signal'] = 1
    else:
        df.loc[base_signal, 'Signal'] = 1

    # -------------------------------------------------------------
    # 5. محاكي Backtest الدقيق
    # -------------------------------------------------------------
    trades = []
    in_pos = False
    entry_price, entry_time = 0, None

    for i in range(len(df)):
        c_price = df['Close'].iloc[i]
        c_time = df.index[i]

        if not in_pos and df['Signal'].iloc[i] == 1:
            in_pos = True
            entry_price = c_price
            entry_time = c_time
        elif in_pos:
            ret = (c_price - entry_price) / entry_price
            if ret >= take_profit_pct or ret <= -stop_loss_pct:
                trades.append({
                    'وقت الدخول': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'وقت الخروج': c_time.strftime('%Y-%m-%d %H:%M'),
                    'سعر الدخول': entry_price,
                    'سعر الخروج': c_price,
                    'العائد %': ret * 100,
                    'النتيجة': 'ربح 🟢' if ret > 0 else 'خسارة 🔴'
                })
                in_pos = False

    tdf = pd.DataFrame(trades)

    # -------------------------------------------------------------
    # 6. العرض والواجهة الرئيسية (Dashboard Layout)
    # -------------------------------------------------------------
    
    # رأس الصفحة والعنوان
    st.title(f"📈 غرفة تداول: {ticker_input} ({interval_selected})")
    
    # بطاقة التوصية اللحظية للحجم الحالي
    latest_bar = df.iloc[-1]
    latest_signal = df['Signal'].iloc[-1]
    
    if latest_signal == 1:
        st.markdown(f"""
        <div class="signal-card-buy">
            <h3 style="color:#2ea043; margin:0;">🚨 إشارة شراء قائمة الآن (BUY / CALL)</h3>
            <p style="margin:5px 0;">السعر الحالي: <b>${latest_bar['Close']:.2f}</b> | الهدف المتوقع: <b>${latest_bar['Close']*(1+take_profit_pct):.2f}</b> | وقف الخسارة: <b>${latest_bar['Close']*(1-stop_loss_pct):.2f}</b></p>
            <small>تم توثيق الإشارة بناءً على {strategy_choice} وفلاتر السيولة.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-card-neutral">
            <h4 style="color:#8b949e; margin:0;">⏳ حالة السوق: المراقبة وانتظار اكتمال شروط الاختراق</h4>
            <p style="margin:5px 0; font-size:0.9rem;">السعر الحالي: <b>${latest_bar['Close']:.2f}</b> | خط الـ VWAP: <b>${latest_bar['VWAP']:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # صف الإحصائيات Key Metrics Row
    if not tdf.empty:
        total_t = len(tdf)
        wins = len(tdf[tdf['النتيجة'] == 'ربح 🟢'])
        win_rate = (wins / total_t) * 100
        p_factor = tdf[tdf['العائد %'] > 0]['العائد %'].sum() / abs(tdf[tdf['العائد %'] < 0]['العائد %'].sum() or 1)
        avg_ret = tdf['العائد %'].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-card"><div class="metric-title">إجمالي الصفقات</div><div class="metric-value">{total_t}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">نسبة النجاح</div><div class="metric-value" style="color:#2ea043;">{win_rate:.1f}%</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Profit Factor</div><div class="metric-value">{p_factor:.2f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-title">متوسط الصفقة</div><div class="metric-value">{avg_ret:.2f}%</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-card"><div class="metric-title">معدل Risk/Reward</div><div class="metric-value">1:{(take_profit_pct/stop_loss_pct):.1f}</div></div>', unsafe_allow_html=True)
    else:
        st.info("💡 لا توجد صفقات مكتملة ضمن الشروط الحالية لتقييم الإحصائيات. يمكنك تعديل الأهداف أو الفريم.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 7. التبويبات التفاعلية (Tabs Setup)
    # -------------------------------------------------------------
    tab_chart, tab_vol, tab_logs = st.tabs(["📊 شارت التداول والإشارات", "🌊 تحليل السيولة والـ VWAP", "📋 سجل الصفقات التفصيلي"])

    with tab_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

        # الشموع اليابانية
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="السعر", increasing_line_color='#2ea043', decreasing_line_color='#da3633'
        ), row=1, col=1)

        # خط الـ VWAP
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#f1e05a', width=1.5), name="VWAP"), row=1, col=1)

        # خطوط المقاومة
        if "S/R" in strategy_choice:
            fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], line=dict(color='#8b949e', width=1, dash='dot'), name="المقاومة"), row=1, col=1)

        # أسهم الشراء/Call
        signals = df[df['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['Low'] * 0.997, mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#2ea043'), name='إشارة شراء مؤكدة'
        ), row=1, col=1)

        # الفوليوم
        vol_colors = ['#2ea043' if c >= o else '#da3633' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, opacity=0.6, name="الحجم"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=600,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_vol:
        st.subheader("تفاصيل تدفق السيولة والـ Volume Spikes")
        st.write("الشموع الملونة باللون الفسفوري تشير إلى وجود سيولة مؤسسية متدفقة تجاوزت المتوسط المحدد.")
        
        fig_vol = go.Figure()
        spike_colors = np.where(df['Volume_Spike'], '#00f5ff', '#30363d')
        fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=spike_colors, name="Vol"))
        fig_vol.add_trace(go.Scatter(x=df.index, y=df['Vol_MA'] * vol_multiplier, line=dict(color='orange', width=1.5), name="عتبة الاختراق"))
        fig_vol.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab_logs:
        st.subheader("سجل الأداء التاريخي")
        if not tdf.empty:
            st.dataframe(tdf.style.format({'سعر الدخول': '${:.2f}', 'سعر الخروج': '${:.2f}', 'العائد %': '{:.2f}%'}), use_container_width=True)
        else:
            st.write("لا توجد صفقات منفذة بعد.")
