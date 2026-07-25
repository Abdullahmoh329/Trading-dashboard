import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# 1. تهيئة واجهة المستخدم Visual Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Pro Quant Terminal & Pattern Detector", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px; text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; }
    .metric-val { font-size: 1.3rem; font-weight: bold; color: #f0f6fc; }
    .call-banner { background: rgba(46, 160, 67, 0.15); border: 2px solid #2ea043; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
    .put-banner { background: rgba(218, 54, 51, 0.15); border: 2px solid #da3633; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
    .wait-banner { background: rgba(139, 148, 158, 0.15); border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 15px; }
    .pattern-badge { background: #21262d; border: 1px solid #58a6ff; color: #58a6ff; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. القائمة الجانبية وإعدادات الباك تست
# -------------------------------------------------------------
st.sidebar.title("⚡ Quant Terminal")
st.sidebar.caption("محرك تحليل الاختراقات والأنماط وحساب الباك تست")

ticker_symbol = st.sidebar.text_input("رمز السهم:", value="AAPL").upper().strip()
timeframe = st.sidebar.selectbox("الفريم الزمني:", ["5m", "15m", "1h", "1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.header("🧪 إعدادات إدارة المخاطر والباك تست")
target_pct = st.sidebar.slider("هدف الربح (Take Profit %):", 0.5, 5.0, 1.5, 0.1) / 100
stop_pct = st.sidebar.slider("وقف الخسارة (Stop Loss %):", 0.3, 3.0, 0.8, 0.1) / 100
vol_multiplier = st.sidebar.slider("مضاعف الفوليوم المطلوب:", 1.1, 2.5, 1.3, 0.1)

# -------------------------------------------------------------
# 3. خوارزمية كشف النماذج الفنية (Pattern Detection Engine)
# -------------------------------------------------------------
def detect_chart_patterns(df):
    patterns = []
    if len(df) < 30:
        return patterns

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values

    # أ) كشف نموذج القمتين المزدوجتين (Double Top) - إشارة انعكاسية هابطة
    recent_highs = df['High'].iloc[-30:]
    max1_idx = recent_highs.iloc[:-10].idxmax()
    max2_idx = recent_highs.iloc[-10:].idxmax()
    max1_val = df.loc[max1_idx, 'High']
    max2_val = df.loc[max2_idx, 'High']
    
    # إذا كانت القمتان متقاربتين بنسبة أقل من 0.4% وكان هناك قاع بينهما
    if abs(max1_val - max2_val) / max1_val < 0.004 and max1_idx != max2_idx:
        mid_low = df.loc[max1_idx:max2_idx, 'Low'].min()
        if (max1_val - mid_low) / max1_val > 0.008:
            patterns.append({
                "name": "⚠️ قمتين مزدوجتين (Double Top)",
                "type": "BEARISH",
                "desc": f"تم كشف قمتين متقاربتين عند ${max1_val:.2f} و ${max2_val:.2f}. تحذير من انعكاس هابط!"
            })

    # ب) كشف نموذج القاعين المزدوجين (Double Bottom) - إشارة انعكاسية صاعدة
    recent_lows = df['Low'].iloc[-30:]
    min1_idx = recent_lows.iloc[:-10].idxmin()
    min2_idx = recent_lows.iloc[-10:].idxmin()
    min1_val = df.loc[min1_idx, 'Low']
    min2_val = df.loc[min2_idx, 'Low']

    if abs(min1_val - min2_val) / min1_val < 0.004 and min1_idx != min2_idx:
        mid_high = df.loc[min1_idx:min2_idx, 'High'].max()
        if (mid_high - min1_val) / min1_val > 0.008:
            patterns.append({
                "name": "🚀 قاعين مزدوجين (Double Bottom)",
                "type": "BULLISH",
                "desc": f"تم كشف قاعين متطابقين عند ${min1_val:.2f}. دلالة على ارتداد صاعد قوي."
            })

    # ج) كشف شمعة ابتلاعية صاعدة (Bullish Engulfing)
    if closes[-2] < opens[-2] and closes[-1] > opens[-1] and closes[-1] > opens[-2] and opens[-1] < closes[-2]:
        patterns.append({
            "name": "🕯️ شمعة ابتلاعية شرائية (Bullish Engulfing)",
            "type": "BULLISH",
            "desc": "الشمعة الحالية ابتلعت الشمعة البيعية السابقة بالكامل مع سيولة دخول."
        })

    # د) كشف شمعة ابتلاعية بيعية (Bearish Engulfing)
    if closes[-2] > opens[-2] and closes[-1] < opens[-1] and closes[-1] < opens[-2] and opens[-1] > closes[-2]:
        patterns.append({
            "name": "🕯️ شمعة ابتلاعية بيعية (Bearish Engulfing)",
            "type": "BEARISH",
            "desc": "الشمعة الحالية ابتلعت الشمعة الشرائية السابقة بالكامل."
        })

    return patterns

# -------------------------------------------------------------
# 4. محرك الباك تست (Backtesting Simulation Engine)
# -------------------------------------------------------------
def run_backtest_simulation(df, tp_rate, sl_rate):
    trades = []
    in_trade = False
    entry_price = 0
    entry_time = None
    trade_type = ""

    for i in range(20, len(df)):
        c_price = df['Close'].iloc[i]
        c_time = df.index[i]
        signal = df['Signal'].iloc[i]

        if not in_trade:
            if signal == 1: # دخول Call
                in_trade = True
                entry_price = c_price
                entry_time = c_time
                trade_type = "CALL 📈"
            elif signal == -1: # دخول Put
                in_trade = True
                entry_price = c_price
                entry_time = c_time
                trade_type = "PUT 📉"
        else:
            if trade_type == "CALL 📈":
                pnl = (c_price - entry_price) / entry_price
                if pnl >= tp_rate or pnl <= -sl_rate:
                    trades.append({
                        "النوع": trade_type,
                        "وقت الدخول": entry_time.strftime("%m-%d %H:%M"),
                        "وقت الخروج": c_time.strftime("%m-%d %H:%M"),
                        "سعر الدخول": entry_price,
                        "سعر الخروج": c_price,
                        "الربح/الخسارة %": round(pnl * 100, 2),
                        "النتيجة": "ربح 🟢" if pnl > 0 else "خسارة 🔴"
                    })
                    in_trade = False
            elif trade_type == "PUT 📉":
                pnl = (entry_price - c_price) / entry_price
                if pnl >= tp_rate or pnl <= -sl_rate:
                    trades.append({
                        "النوع": trade_type,
                        "وقت الدخول": entry_time.strftime("%m-%d %H:%M"),
                        "وقت الخروج": c_time.strftime("%m-%d %H:%M"),
                        "سعر الدخول": entry_price,
                        "سعر الخروج": c_price,
                        "الربح/الخسارة %": round(pnl * 100, 2),
                        "النتيجة": "ربح 🟢" if pnl > 0 else "خسارة 🔴"
                    })
                    in_trade = False

    return pd.DataFrame(trades)

# -------------------------------------------------------------
# 5. جلب البيانات وتحليل السوق المباشر
# -------------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_prep_data(symbol, interval):
    df = yf.download(symbol, period="1mo" if interval in ["5m", "15m"] else "1y", interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

df = load_and_prep_data(ticker_symbol, timeframe)

if df.empty or len(df) < 30:
    st.error(f"⚠️ تعذر جلب بيانات السهم '{ticker_symbol}'. يرجى التأكد من الرمز.")
else:
    # حساب المؤشرات
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']

    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] > (df['Vol_MA'] * vol_multiplier)

    df['Resistance'] = df['High'].rolling(20).max().shift(1)
    df['Support'] = df['Low'].rolling(20).min().shift(1)

    # توليد الإشارات بناءً على الاختراق وإعادة الاختبار والـ VWAP
    df['Signal'] = 0
    
    # شروط الـ CALL
    call_cond = (df['Close'] > df['VWAP']) & (df['EMA_9'] > df['EMA_21']) & (df['Vol_Spike'])
    # شروط الـ PUT
    put_cond = (df['Close'] < df['VWAP']) & (df['EMA_9'] < df['EMA_21']) & (df['Vol_Spike'])

    df.loc[call_cond, 'Signal'] = 1
    df.loc[put_cond, 'Signal'] = -1

    # فحص الأنماط الفنية
    detected_patterns = detect_chart_patterns(df)

    # تقييم القرار اللحظي بناءً على المؤشرات والأنماط المكتشفة
    latest = df.iloc[-1]
    curr_price = float(latest['Close'])
    vwap_val = float(latest['VWAP'])

    score_call = 0
    score_put = 0

    if curr_price > vwap_val: score_call += 30
    else: score_put += 30

    if latest['EMA_9'] > latest['EMA_21']: score_call += 30
    else: score_put += 30

    # خصم/إضافة نقاط بناءً على كشف الأنماط (حل مشكلة خديعة الـ Double Top)
    for p in detected_patterns:
        if p['type'] == 'BEARISH':
            score_put += 40
            score_call -= 30 # إضعاف إشارة الـ CALL فوراً
        elif p['type'] == 'BULLISH':
            score_call += 40
            score_put -= 30

    # القرار النهائي
    if score_put >= 60:
        action = "STRONG PUT 📉 (توصية هابطة / شراء عقود Put)"
        banner = "put-banner"
        color = "#da3633"
    elif score_call >= 60:
        action = "STRONG CALL 📈 (توصية صاعدة / شراء عقود Call)"
        banner = "call-banner"
        color = "#2ea043"
    else:
        action = "WAIT / NEUTRAL ⏳ (انتظار - عدم الدخول)"
        banner = "wait-banner"
        color = "#8b949e"

    # -------------------------------------------------------------
    # 6. العرض الرئيسي والتبويبات
    # -------------------------------------------------------------
    st.title(f"📈 غرفة تداول: {ticker_symbol} ({timeframe})")

    st.markdown(f"""
    <div class="{banner}">
        <h2 style="margin:0; color:{color};">{action}</h2>
        <p style="margin:5px 0 0 0;">السعر الحالي: <b>${curr_price:.2f}</b> | الـ VWAP: <b>${vwap_val:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # تبويبات التنقل
    tab_chart, tab_patterns, tab_backtest = st.tabs([
        "📊 الرسم البياني والإشارات", 
        "🔍 كاشف النماذج الفنية (Patterns)", 
        "🧪 محاكي الباك تست (Backtest)"
    ])

    with tab_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="السعر"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#f1e05a', width=1.5), name="VWAP"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='#58a6ff', width=1), name="EMA 9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='#bc8cff', width=1), name="EMA 21"), row=1, col=1)
        
        # الفوليوم
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='#30363d', name="Volume"), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=550, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_patterns:
        st.subheader("🔍 الأنماط الفنية والشمعية المكتشفة حيوياً:")
        if detected_patterns:
            for pat in detected_patterns:
                p_color = "#da3633" if pat['type'] == 'BEARISH' else "#2ea043"
                st.markdown(f"""
                <div style="background:#161b22; border-right:4px solid {p_color}; padding:12px; margin-bottom:10px; border-radius:6px;">
                    <h4 style="margin:0; color:{p_color};">{pat['name']}</h4>
                    <p style="margin:4px 0 0 0; color:#c9d1d9;">{pat['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("لم يتم رصد أنماط فنية معقدة (Double Top/Bottom) على الشموع الأخيرة حالياً.")

    with tab_backtest:
        st.subheader("🧪 نتائج محاكاة الباك تست (Backtest Engine)")
        
        trades_df = run_backtest_simulation(df, target_pct, stop_pct)
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            wins = len(trades_df[trades_df['النتيجة'] == 'ربح 🟢'])
            win_rate = (wins / total_trades) * 100
            total_return = trades_df['الربح/الخسارة %'].sum()
            profit_factor = trades_df[trades_df['الربح/الخسارة %'] > 0]['الربح/الخسارة %'].sum() / abs(trades_df[trades_df['الربح/الخسارة %'] < 0]['الربح/الخسارة %'].sum() or 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">إجمالي الصفقات المنفذة</div><div class="metric-val">{total_trades}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-label">نسبة النجاح (Win Rate)</div><div class="metric-val" style="color:#2ea043;">{win_rate:.1f}%</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-label">صافي العائد التراكمي</div><div class="metric-val" style="color:#58a6ff;">{total_return:.2f}%</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-label">معامل الربحية (Profit Factor)</div><div class="metric-val">{profit_factor:.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("📋 **سجل الصفقات التاريخي التفصيلي:**")
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.warning("لا توجد صفقات منفذة ضمن شروط الباك تست الحالية على هذا الفريم. يمكنك تغيير نسبة الهدف أو وقف الخسارة من الشريط الجانبي.")
