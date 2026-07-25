import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Intraday Multi-Strategy Quant Engine", layout="wide")

st.title("⚡ Intraday Multi-Strategy Quant Engine (5m / 15m)")
st.caption("اختبار استراتيجيات السكالبينغ، اختراق الترندات، والدعم والمقاومة مع تأكيد إعادة الاختبار (Retest)")

# القائمة الجانبية
st.sidebar.header("⚙️ إعدادات التداول والبيانات")
ticker_input = st.sidebar.text_input("رمز السهم (مثل NVDA, AMD, TSLA):", value="NVDA").upper().strip()

interval_selected = st.sidebar.selectbox("الفريم الزمني (Timeframe):", ["5m", "15m", "1d"], index=1)

# ضبط فترة البيانات بناءً على الفريم (Yahoo Finance يحدد 1 mo للفريمات السريعة)
period_map = {"5m": "1mo", "15m": "1mo", "1d": "2y"}
period_selected = period_map[interval_selected]

strategy_choice = st.sidebar.selectbox(
    "اختر الاستراتيجية للاختبار الإحصائي:",
    [
        "اختراق الدعم/المقاومة + إعادة الاختبار (S/R Retest)",
        "اختراق خط الترند الديناميكي (Trendline Breakout)",
        "تقاطع المتوسطات السريعة (EMA Scalper 9/21)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 إعدادات إدارة المخاطر")
stop_loss_pct = st.sidebar.slider("وقف الخسارة (Stop Loss %):", 0.5, 3.0, 1.0, 0.1) / 100
take_profit_pct = st.sidebar.slider("هدف الربح (Take Profit %):", 1.0, 6.0, 2.0, 0.1) / 100

# جلب البيانات
@st.cache_data(ttl=300)
def fetch_intraday_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except:
        return pd.DataFrame()

df = fetch_intraday_data(ticker_input, period_selected, interval_selected)

if df.empty or len(df) < 50:
    st.error(f"⚠️ تعذر جلب بيانات الفريم الزمني المطلوب للسهم '{ticker_input}'.")
else:
    # -------------------------------------------------------------
    # 1. خوارزميات الاستراتيجيات
    # -------------------------------------------------------------
    df['Signal'] = 0 # 1 = Buy (Call), -1 = Sell (Put), 0 = Neutral
    
    lookback = 20 # عدد الشموع لتحديد القمم والقيعان
    
    if strategy_choice == "اختراق الدعم/المقاومة + إعادة الاختبار (S/R Retest)":
        # تحديد المقاومة والدعم السابقيين
        df['Resistance'] = df['High'].rolling(window=lookback).max().shift(1)
        df['Support'] = df['Low'].rolling(window=lookback).min().shift(1)
        
        # شرط الاختراق + إعادة الاختبار:
        # 1. كسر المقاومة سابقاً
        # 2. عودة السعر لأقرب نقطة من المقاومة المكسورة (أصبحت دعم)
        # 3. إغلاق شمعة صاعدة لتأكيد الارتداد
        
        breakout = df['Close'].shift(1) > df['Resistance'].shift(1)
        retest = (df['Low'] <= df['Resistance'] * 1.002) & (df['Close'] > df['Resistance'])
        bullish_candle = df['Close'] > df['Open']
        
        df.loc[breakout & retest & bullish_candle, 'Signal'] = 1
        
    elif strategy_choice == "اختراق خط الترند الديناميكي (Trendline Breakout)":
        # حساب متوسط القمم والانحدار لتحديد الترند الهابط
        df['Rolling_Max'] = df['High'].rolling(window=10).max()
        df['Trendline'] = df['Rolling_Max'].shift(1)
        
        # إشارة شراء عند اختراق خط القمم السابقة بكتلة فوليوم أعلى من المتوسط
        vol_ma = df['Volume'].rolling(20).mean()
        df.loc[(df['Close'] > df['Trendline']) & (df['Volume'] > vol_ma), 'Signal'] = 1
        
    elif strategy_choice == "تقاطع المتوسطات السريعة (EMA Scalper 9/21)":
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        cross_up = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1))
        df.loc[cross_up, 'Signal'] = 1

    # -------------------------------------------------------------
    # 2. محرك محاكاة الصفقات (Backtest Simulator Execution)
    # -------------------------------------------------------------
    trades = []
    in_position = False
    entry_price = 0
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_time = df.index[i]
        
        if not in_position and df['Signal'].iloc[i] == 1:
            in_position = True
            entry_price = current_price
            entry_time = current_time
            
        elif in_position:
            price_change = (current_price - entry_price) / entry_price
            
            # تحقق من الهدف أو وقف الخسارة
            if price_change >= take_profit_pct or price_change <= -stop_loss_pct:
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Return %': price_change * 100,
                    'Result': 'WIN' if price_change > 0 else 'LOSS'
                })
                in_position = False

    trades_df = pd.DataFrame(trades)

    # -------------------------------------------------------------
    # 3. عرض النتائج والإحصائيات الإشارات
    # -------------------------------------------------------------
    st.subheader(f"📊 نتائج باك تيست استراتيجية: [{strategy_choice}]")
    st.caption(f"السهم: **{ticker_input}** | الفريم: **{interval_selected}** | الهدف: **+{take_profit_pct*100:.1f}%** | الوقف: **-{stop_loss_pct*100:.1f}%**")

    if trades_df.empty:
        st.warning("⚠️ لم يتم رصد صفقات المكتملة ضمن الفترة الزمنية المحددة بهذه الشروط. جرب تقليل شروط الوقف/الهدف أو تغيير الفريم.")
    else:
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['Result'] == 'WIN'])
        loss_trades = len(trades_df[trades_df['Result'] == 'LOSS'])
        win_rate = (win_trades / total_trades) * 100
        
        total_profit = trades_df[trades_df['Result'] == 'WIN']['Return %'].sum()
        total_loss = abs(trades_df[trades_df['Result'] == 'LOSS']['Return %'].sum())
        profit_factor = (total_profit / total_loss) if total_loss > 0 else total_profit

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("إجمالي الصفقات (Trades)", f"{total_trades}")
        c2.metric("نسبة النجاح (Win Rate)", f"{win_rate:.1f}%")
        c3.metric("معامل الأرباح (Profit Factor)", f"{profit_factor:.2f}")
        c4.metric("متوسط أداء الصفقة", f"{trades_df['Return %'].mean():.2f}%")

        # رسم البياني مع مناطق الدخول
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="السعر"
        ))
        
        # إضافة نقاط الإشارات
        buy_signals = df[df['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Low'] * 0.998,
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00FF7F'),
            name='إشارة دخول (Buy/Call)'
        ))

        fig.update_layout(
            title=f"شارت {ticker_input} ({interval_selected}) - إشارات الاستراتيجية",
            template="plotly_dark", height=500, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # جدول الصفقات التاريخية
        st.subheader("📋 سجل الصفقات التفصيلي")
        st.dataframe(trades_df.tail(15).style.format({
            'Entry Price': '{:.2f}', 'Exit Price': '{:.2f}', 'Return %': '{:.2f}%'
        }), use_container_width=True)
