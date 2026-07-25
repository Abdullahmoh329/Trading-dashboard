import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# -------------------------------------------------------------
# 1. Modern Dark Terminal UI Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Quant Options Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .terminal-header {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 2px solid #58a6ff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
    }
    .badge-recommend {
        background: #238636;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-label { font-size: 0.82rem; color: #8b949e; margin-bottom: 6px; font-weight: 500; }
    .stat-val-green { font-size: 1.35rem; font-weight: 700; color: #3fb950; }
    .stat-val-red { font-size: 1.35rem; font-weight: 700; color: #f85149; }
    .stat-val-blue { font-size: 1.35rem; font-weight: 700; color: #58a6ff; }
    
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Robust Market & Options Data Fetcher
# -------------------------------------------------------------
def get_custom_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    })
    return session

@st.cache_data(ttl=120)
def load_market_data(symbol_str):
    try:
        session = get_custom_session()
        ticker = yf.Ticker(symbol_str, session=session)
        
        hist = ticker.history(period="1mo", interval="15m")
        if hist.empty:
            return None, [], None, 0.0

        current_price = float(hist['Close'].iloc[-1])

        # Automatic ATR Calculation for Target/Stop Loss
        high_low = hist['High'] - hist['Low']
        high_cp = np.abs(hist['High'] - hist['Close'].shift())
        low_cp = np.abs(hist['Low'] - hist['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr_val = float(tr.rolling(14).mean().iloc[-1])

        # Expiration dates fetch with fallback
        try:
            expirations = list(ticker.options)
        except Exception:
            expirations = []

        return current_price, expirations, hist, atr_val
    except Exception:
        return None, [], None, 0.0

@st.cache_data(ttl=120)
def load_options_chain(symbol_str, exp_date, is_call_mode):
    try:
        session = get_custom_session()
        ticker = yf.Ticker(symbol_str, session=session)
        chain = ticker.option_chain(exp_date)
        return chain.calls if is_call_mode else chain.puts
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------------------
# 3. Sidebar Setup
# -------------------------------------------------------------
st.sidebar.markdown("## ⚡ Quant Terminal")
symbol = st.sidebar.text_input("Stock Ticker:", value="AMD").upper().strip()

live_price, expirations, price_hist, atr_value = load_market_data(symbol)

if not live_price:
    st.sidebar.error("⚠️ Connection Error")
    if st.sidebar.button("🔄 Retry Connection"):
        st.cache_data.clear()
        st.rerun()
    st.error(f"Failed to fetch market data for **{symbol}**. Click 'Retry Connection' in the sidebar.")
else:
    st.sidebar.markdown(f"**Live Stock Price:** `${live_price:.2f}`")
    st.sidebar.markdown(f"**Volatility (14-ATR):** `${atr_value:.2f}`")
    st.sidebar.markdown("---")

    trade_type = st.sidebar.radio("Direction Strategy:", ["CALL (Bullish) 📈", "PUT (Bearish) 📉"])
    is_call = "CALL" in trade_type

    # Automated Target and Stop Loss using ATR (No manual input needed)
    if is_call:
        auto_tp = round(live_price + (1.5 * atr_value), 2)
        auto_sl = round(live_price - (1.0 * atr_value), 2)
    else:
        auto_tp = round(live_price - (1.5 * atr_value), 2)
        auto_sl = round(live_price + (1.0 * atr_value), 2)

    st.sidebar.markdown("### 🤖 Auto-Calculated Targets")
    st.sidebar.info(f"**Target Price (TP):** ${auto_tp}\n\n**Stop Loss (SL):** ${auto_sl}")

    if expirations:
        selected_exp = st.sidebar.selectbox("Select Expiration Date:", expirations[:8])
    else:
        st.sidebar.warning("⚠️ Options chain currently unavailable for this ticker.")
        selected_exp = None

    # -------------------------------------------------------------
    # 4. Quant Chain Analyzer Engine
    # -------------------------------------------------------------
    def analyze_options_chain(opts_df, current_s, tp_s, sl_s, call_mode):
        if opts_df.empty:
            return None

        opts = opts_df[(opts_df['strike'] >= current_s * 0.85) & (opts_df['strike'] <= current_s * 1.15)].copy()
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
                "strike": strike, "ask": ask, "bid": bid, "volume": int(volume),
                "delta": round(est_delta, 2), "opt_tp": round(opt_tp_price, 2),
                "opt_sl": round(opt_sl_price, 2), "roi": round(roi_pct, 1),
                "risk": round(risk_pct, 1), "rr": rr_ratio, "score": score
            })

        df_res = pd.DataFrame(results)
        return df_res.sort_values(by="score", ascending=False) if not df_res.empty else None

    # -------------------------------------------------------------
    # 5. Dashboard Rendering
    # -------------------------------------------------------------
    st.markdown(f"""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin:0; color: #f0f6fc;">Options Quant Dashboard: <span style="color:#58a6ff;">{symbol}</span></h2>
                <p style="margin:4px 0 0 0; color:#8b949e;">Algorithmic Strike Selector & Risk Evaluator</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: #8b949e;">Stock Price</span>
                <div style="font-size: 1.6rem; font-weight: bold; color: #f0f6fc;">${live_price:.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if selected_exp:
        raw_opts = load_options_chain(symbol, selected_exp, is_call)
        options_df = analyze_options_chain(raw_opts, live_price, auto_tp, auto_sl, is_call)

        if options_df is not None and not options_df.empty:
            best = options_df.iloc[0]

            st.markdown(f"""
            <div class="hero-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="badge-recommend">🏆 OPTIMAL CONTRACT CHOICE</span>
                    <span style="color: #8b949e; font-size: 0.85rem;">EXP: <b>{selected_exp}</b></span>
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #f0f6fc; margin-bottom: 8px;">
                    {symbol} ${best['strike']:.1f} {'CALL' if is_call else 'PUT'}
                </div>
                <p style="margin:0; color: #8b949e; font-size: 0.95rem;">
                    Buy Price (Ask): <b style="color:#f0f6fc;">${best['ask']:.2f}</b> (${best['ask']*100:.0f}/contract) | 
                    Volume: <b style="color:#f0f6fc;">{best['volume']:,}</b> | 
                    Estimated Delta: <b style="color:#f0f6fc;">{best['delta']}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="stat-card"><div class="stat-label">Option Target (TP)</div><div class="stat-val-green">${best["opt_tp"]:.2f} (+{best["roi"]}%)</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="stat-card"><div class="stat-label">Option Stop Loss (SL)</div><div class="stat-val-red">${best["opt_sl"]:.2f} (-{best["risk"]}%)</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="stat-card"><div class="stat-label">Risk/Reward Ratio</div><div class="stat-val-blue">1:{best["rr"]}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="stat-card"><div class="stat-label">Expected Profit / Contract</div><div class="stat-val-green">+${(best["opt_tp"] - best["ask"])*100:.0f}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab_chart, tab_table = st.tabs(["📈 Price Chart & Targets", "📋 Option Chain Ranking"])

            with tab_chart:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=price_hist.index, open=price_hist['Open'],
                    high=price_hist['High'], low=price_hist['Low'],
                    close=price_hist['Close'], name="Price"
                ))
                fig.add_hline(y=auto_tp, line_dash="dash", line_color="#3fb950", line_width=2, annotation_text=f"Auto TP (${auto_tp})")
                fig.add_hline(y=auto_sl, line_dash="dash", line_color="#f85149", line_width=2, annotation_text=f"Auto SL (${auto_sl})")
                fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab_table:
                clean_df = options_df[['strike', 'ask', 'bid', 'delta', 'opt_tp', 'roi', 'opt_sl', 'risk', 'rr', 'volume']].copy()
                clean_df.columns = ['Strike', 'Ask Price', 'Bid Price', 'Delta', 'Option TP ($)', 'Expected ROI (%)', 'Option SL ($)', 'Max Risk (%)', 'R:R Ratio', 'Volume']
                st.dataframe(clean_df, use_container_width=True, height=380)

        else:
            st.warning("No suitable options contracts found for the selected expiration date.")
