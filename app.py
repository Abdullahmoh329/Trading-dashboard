import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# -------------------------------------------------------------
# 1. تهيئة الصفحة والتصميم Modern Dark Terminal UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="Quant Options Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .terminal-header {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.1) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 2px solid #58a6ff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 0 25px rgba(88, 166, 255, 0.15);
    }
    .badge-recommend {
        background: #238636;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-label { font-size: 0.82rem; color: #8b949e; margin-bottom: 6px; }
    .stat-val-green { font-size: 1.4rem; font-weight: 700; color: #3fb950; }
    .stat-val-red { font-size: 1.4rem; font-weight: 700; color: #f85149; }
    .stat-val-blue { font-size: 1.4rem; font-weight: 700; color: #58a6ff; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. جلب البيانات بدون حظر وبدون أخطاء Unserializable
# -------------------------------------------------------------
def get_custom_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    })
    return session

@st.cache_data(ttl=180) # كاش مسموح لأنه يرجع DataFrames وأرقام فقط (Data-serializable)
def fetch_stock_and_expirations(symbol_str):
    try:
        session = get_custom_session()
        ticker = yf.Ticker(symbol_str, session=session)
        
        hist = ticker.history(period="1mo", interval="15m")
        if hist.empty:
            return None, [], None
        
        current_price = float(hist['Close'].iloc[-1])
        
        try:
            expirations = list(ticker.options)
        except Exception:
            expirations = []
            
        return current_price, expirations, hist
    except Exception:
        return None, [], None

@st.cache_data(ttl=180) # كاش لجلب سلسلة العقود كـ DataFrame صافي
def fetch_options_chain_data(symbol_str, exp_date, call_mode):
    try:
        session = get_custom_session()
        ticker = yf.Ticker(symbol_str, session=session)
        chain = ticker.option_chain(exp_date)
        opts = chain.calls if call_mode else chain.puts
        return opts
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------
# 3. القائمة الجانبية وإدخال البيانات
# -------------------------------------------------------------
st.sidebar.markdown("### ⚡ محرك تداول العقود")
symbol = st.sidebar.text_input("رمز السهم (Ticker):", value="AMD").upper().strip()

live_price, expirations, price_hist = fetch_stock_and_expirations(symbol)

if not live_price:
    st.error(f"⚠️ يتعذر جلب بيانات السهم **{symbol}**. جرب الضغط على Clear Cache من أعلى اليمين أو تأكد من فتح السوق.")
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**سعر السهم اللحظي:** `${live_price:.2f}`")

    trade_type = st.sidebar.radio("نوع التوصية / الصفقة:", ["CALL (صعود) 📈", "PUT (هبوط) 📉"])
    is_call = "CALL" in trade_type

    default_tp = round(live_price * 1.025 if is_call else live_price * 0.975, 2)
    default_sl = round(live_price * 0.990 if is_call else live_price * 1.010, 2)

    st.sidebar.markdown("#### 🎯 المستهدفات الفنية للسهم")
    target_price = st.sidebar.number_input("هدف السهم (Take Profit $):", value=default_tp, step=0.5)
    stop_loss = st.sidebar.number_input("وقف خسارة السهم (Stop Loss $):", value=default_sl, step=0.5)

    st.sidebar.markdown("#### 📅 تاريخ انتهاء الأوبشن")
    selected_exp = st.sidebar.selectbox("اختر التاريخ:", expirations[:6] if expirations else ["غير متاح"])

    # -------------------------------------------------------------
    # 4. محرك تحليل العقود
    # -------------------------------------------------------------
    def analyze_chain(opts_df, current_s, tp_s, sl_s, call_mode):
        if opts_df.empty:
            return None

        opts = opts_df[(opts_df['strike'] >= current_s * 0.82) & (opts_df['strike'] <= current_s * 1.18)].copy()
        opts = opts[opts['ask'] > 0.05].copy()

        if opts.empty:
            return None

        results = []
        price_change_tp = abs(tp_s - current_s)
        price_change_sl = abs(current_s - sl_s)

        for _, row in opts.iterrows():
            strike = row['strike']
            ask = row['ask']
            bid = row['bid']
            volume = row['volume'] if not np.isnan(row['volume']) else 0
            oi = row['openInterest'] if not np.isnan(row['openInterest']) else 0

            moneness = (current_s - strike) if call_mode else (strike - current_s)
            est_delta = min(0.85, max(0.15, 0.50 + (moneness / current_s) * 2.8))

            opt_tp_price = ask + (price_change_tp * est_delta)
            opt_sl_price = max(0.01, ask - (price_change_sl * est_delta))

            opt_profit = opt_tp_price - ask
            opt_loss = ask - opt_sl_price

            roi_pct = (opt_profit / ask) * 100
            risk_pct = (opt_loss / ask) * 100
            rr_ratio = round(opt_profit / opt_loss, 2) if opt_loss > 0 else 0

            score = (roi_pct * 0.45) + (rr_ratio * 15) + (np.log1p(volume) * 2.5)

            results.append({
                "strike": strike,
                "ask": ask,
                "bid": bid,
                "volume": int(volume),
                "openInterest": int(oi),
                "delta": round(est_delta, 2),
                "opt_tp": round(opt_tp_price, 2),
                "opt_sl": round(opt_sl_price, 2),
                "roi": round(roi_pct, 1),
                "risk": round(risk_pct, 1),
                "rr": rr_ratio,
                "score": score
            })

        df_res = pd.DataFrame(results)
        return df_res.sort_values(by="score", ascending=False) if not df_res.empty else None

    # -------------------------------------------------------------
    # 5. العرض والتصميم الرئيسي
    # -------------------------------------------------------------
    st.markdown(f"""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin:0; font-size: 1.8rem; color: #f0f6fc;">غرفة صفقات الأوبشن: <span style="color:#58a6ff;">{symbol}</span></h1>
                <p style="margin:5px 0 0 0; color:#8b949e;">تحليل حظي لسلسلة العقود وتحديد السترايك الأفضل</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: #8b949e;">السعر اللحظي</span>
                <div style="font-size: 1.6rem; font-weight: bold; color: #f0f6fc;">${live_price:.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # جلب بيانات الأوبشن
    raw_opts = fetch_options_chain_data(symbol, selected_exp, is_call)
    options_df = analyze_chain(raw_opts, live_price, target_price, stop_loss, is_call)

    if options_df is not None and not options_df.empty:
        best = options_df.iloc[0]

        st.markdown(f"""
        <div class="hero-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="badge-recommend">🏆 العقد الأفضل الموصى به</span>
                <span style="color: #8b949e; font-size: 0.85rem;">تاريخ الانتهاء: <b>{selected_exp}</b></span>
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: #f0f6fc; margin-bottom: 8px;">
                {symbol} ${best['strike']:.1f} {'CALL' if is_call else 'PUT'}
            </div>
            <p style="margin:0; color: #8b949e; font-size: 0.9rem;">
                سعر شراء العقد (Ask): <b style="color:#f0f6fc;">${best['ask']:.2f}</b> (${best['ask']*100:.0f} لكل عقد) | 
                السيولة (Volume): <b style="color:#f0f6fc;">{best['volume']:,}</b> | 
                الدلتا التقديرية: <b style="color:#f0f6fc;">{best['delta']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="stat-card"><div class="stat-label">سعر بيع العقد المستهدف (TP)</div><div class="stat-val-green">${best["opt_tp"]:.2f} (+{best["roi"]}%)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="stat-card"><div class="stat-label">سعر وقف خسارة العقد (SL)</div><div class="stat-val-red">${best["opt_sl"]:.2f} (-{best["risk"]}%)</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="stat-card"><div class="stat-label">نسبة العائد للمخاطرة (R:R)</div><div class="stat-val-blue">1:{best["rr"]}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="stat-card"><div class="stat-label">الربح الصافي المتوقع / عقد</div><div class="stat-val-green">+${(best["opt_tp"] - best["ask"])*100:.0f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab_chart, tab_table = st.tabs(["📈 الرسم البياني والمستهدفات", "📋 جدول مفاضلة باقي العقود"])

        with tab_chart:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_hist.index,
                open=price_hist['Open'], high=price_hist['High'],
                low=price_hist['Low'], close=price_hist['Close'], name="السعر"
            ))
            fig.add_hline(y=target_price, line_dash="dash", line_color="#3fb950", line_width=2, annotation_text=f"الهدف TP (${target_price})")
            fig.add_hline(y=stop_loss, line_dash="dash", line_color="#f85149", line_width=2, annotation_text=f"الوقف SL (${stop_loss})")

            fig.update_layout(
                template="plotly_dark", height=450,
                paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_table:
            clean_df = options_df[['strike', 'ask', 'bid', 'delta', 'opt_tp', 'roi', 'opt_sl', 'risk', 'rr', 'volume']].copy()
            clean_df.columns = [
                'السترايك (Strike)', 'سعر الشراء (Ask)', 'سعر البيع (Bid)', 
                'Delta', 'هدف العقد ($)', 'الربح المتوقع (%)', 
                'وقف العقد ($)', 'المخاطرة (%)', 'العائد/المخاطرة', 'حجم التداول'
            ]
            st.dataframe(clean_df, use_container_width=True, height=400)

    else:
        st.warning("لم يتم العثور على عقود تلبّي الشروط للتاريخ المحدد. اختر تاريخ انتهاء آخر من القائمة الجانبية.")
