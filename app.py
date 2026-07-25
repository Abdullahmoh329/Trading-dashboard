import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# -------------------------------------------------------------
# 1. Page Configuration & Professional Dark CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="QuantVision Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #161b22;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-size: 1rem;
        font-weight: 600;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
    }

    /* Cards & Containers */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; }
    .metric-val-green { font-size: 1.4rem; font-weight: 800; color: #3fb950; }
    .metric-val-red { font-size: 1.4rem; font-weight: 800; color: #f85149; }
    .metric-val-blue { font-size: 1.4rem; font-weight: 800; color: #58a6ff; }
    
    .hero-banner {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.15) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid #1f6feb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Resilient Data Engine with Secure Session Headers
# -------------------------------------------------------------
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    return session

@st.cache_data(ttl=60)
def fetch_stock_data(ticker_symbol, timeframe="6m"):
    df = pd.DataFrame()
    try:
        session = get_session()
        ticker = yf.Ticker(ticker_symbol, session=session)
        df = ticker.history(period=timeframe)
        
        if df is None or df.empty:
            df = yf.download(ticker_symbol, period=timeframe, progress=False, session=session)
            
        if df is None or df.empty:
            return None, 0.0, 0.0
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        curr_price = float(df['Close'].iloc[-1])
        return df, curr_price, atr
    except Exception:
        return None, 0.0, 0.0

def fetch_options_chain(ticker_symbol):
    try:
        session = get_session()
        ticker = yf.Ticker(ticker_symbol, session=session)
        expirations = list(ticker.options)
        return ticker, expirations
    except Exception:
        return None, []

# -------------------------------------------------------------
# 3. Pattern Recognition Logic
# -------------------------------------------------------------
def detect_patterns(df):
    patterns = []
    if len(df) < 30:
        return ["Insufficient data history for pattern detection"]

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    r_closes = closes[-20:]
    r_highs = highs[-20:]
    r_lows = lows[-20:]

    min1_idx = np.argmin(r_lows[:10])
    min2_idx = np.argmin(r_lows[10:]) + 10
    if abs(r_lows[min1_idx] - r_lows[min2_idx]) / r_lows[min1_idx] < 0.018:
        patterns.append("🟢 Potential Double Bottom (Bullish Reversal)")

    max1_idx = np.argmax(r_highs[:10])
    max2_idx = np.argmax(r_highs[10:]) + 10
    if abs(r_highs[max1_idx] - r_highs[max2_idx]) / r_highs[max1_idx] < 0.018:
        patterns.append("🔴 Potential Double Top (Bearish Reversal)")

    initial_move = (closes[-15] - closes[-30]) / closes[-30]
    recent_range = (max(r_highs[-10:]) - min(r_lows[-10:])) / min(r_lows[-10:])
    if initial_move > 0.04 and recent_range < 0.035:
        patterns.append("🚀 Bull Flag Consolidation")
    elif initial_move < -0.04 and recent_range < 0.035:
        patterns.append("⚠️ Bear Flag Consolidation")

    if not patterns:
        patterns.append("➡️ Market in Standard Consolidation (No Classic Pattern Triggered)")

    return patterns

# -------------------------------------------------------------
# 4. Quantitative Backtesting Engine
# -------------------------------------------------------------
def run_backtest(df, fast_ma, slow_ma, initial_capital=10000):
    data = df.copy()
    data['Fast_MA'] = data['Close'].rolling(window=fast_ma).mean()
    data['Slow_MA'] = data['Close'].rolling(window=slow_ma).mean()
    
    data['Signal'] = 0
    data.iloc[fast_ma:, data.columns.get_loc('Signal')] = np.where(
        data['Fast_MA'].iloc[fast_ma:] > data['Slow_MA'].iloc[fast_ma:], 1, -1
    )
    
    data['Position'] = data['Signal'].shift(1)
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position']
    
    data['Cumulative_Market'] = (1 + data['Market_Returns']).fillna(0).cumprod() * initial_capital
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).fillna(0).cumprod() * initial_capital
    
    total_trades = int((data['Position'].diff().abs() > 0).sum())
    net_profit = float(data['Cumulative_Strategy'].iloc[-1] - initial_capital)
    ret_pct = (net_profit / initial_capital) * 100
    
    winning_days = (data['Strategy_Returns'] > 0).sum()
    total_active_days = (data['Strategy_Returns'] != 0).sum()
    win_rate = (winning_days / total_active_days * 100) if total_active_days > 0 else 0.0
    
    rolling_max = data['Cumulative_Strategy'].cummax()
    drawdown = (data['Cumulative_Strategy'] - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100) if not drawdown.empty else 0.0
    
    return data, {
        "net_profit": net_profit,
        "ret_pct": ret_pct,
        "win_rate": win_rate,
        "max_dd": max_drawdown,
        "trades": total_trades
    }

# -------------------------------------------------------------
# 5. Application Controls & Sidebar
# -------------------------------------------------------------
st.sidebar.markdown("## ⚡ Quant Terminal")
symbol = st.sidebar.text_input("Asset Ticker Symbol:", value="NVDA").upper().strip()

stock_df, live_price, atr_val = fetch_stock_data(symbol)

if stock_df is None:
    st.sidebar.error("⚠️ Data connection delayed.")
    if st.sidebar.button("🔄 Reload / Retry Connection"):
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown(f"**Current Price:** `${live_price:.2f}`")
st.sidebar.markdown(f"**14-ATR Volatility:** `${atr_val:.2f}`")
st.sidebar.markdown("---")

# Main Interface Navigation Tabs
view_mode = st.tabs(["📈 Stock Analytics & Backtesting", "🎯 Options Quant Selector"])

# -------------------------------------------------------------
# TAB 1: STOCKS, PATTERNS & BACKTESTING
# -------------------------------------------------------------
with view_mode[0]:
    st.markdown(f"""
    <div class="hero-banner">
        <h2 style="margin:0; color:#f0f6fc;">Stock Analytics & Backtesting: <span style="color:#58a6ff;">{symbol}</span></h2>
        <p style="margin:5px 0 0 0; color:#8b949e;">Algorithmic Pattern Recognition Engine & Moving Average Backtest</p>
    </div>
    """, unsafe_allow_html=True)
    
    if stock_df is not None and not stock_df.empty:
        detected_patterns = detect_patterns(stock_df)
        st.markdown("### 🔍 Live Chart Pattern Recognition")
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            for p in detected_patterns:
                st.info(f"**Detected:** {p}")
        with col_p2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pattern Model Status</div>
                <div class="metric-val-blue">Active Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 🧪 Quantitative Strategy Backtest")
        
        b_col1, b_col2, b_col3 = st.columns(3)
        fast_period = b_col1.number_input("Fast MA Period (Days):", min_value=5, max_value=50, value=10)
        slow_period = b_col2.number_input("Slow MA Period (Days):", min_value=20, max_value=200, value=30)
        capital = b_col3.number_input("Initial Capital ($):", min_value=1000, value=10000, step=1000)
        
        bt_df, metrics = run_backtest(stock_df, fast_period, slow_period, capital)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Net Profit</div><div class="{"metric-val-green" if metrics["net_profit"]>=0 else "metric-val-red"}">${metrics["net_profit"]:.2f} ({metrics["ret_pct"]:.1f}%)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-val-blue">{metrics["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-val-red">{metrics["max_dd"]:.1f}%</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">Total Trades</div><div class="metric-val-blue">{metrics["trades"]}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        
        fig_bt.add_trace(go.Candlestick(
            x=bt_df.index, open=bt_df['Open'], high=bt_df['High'], low=bt_df['Low'], close=bt_df['Close'], name="Stock Price"
        ), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Fast_MA'], line=dict(color='#58a6ff', width=1.5), name=f'Fast MA ({fast_period})'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Slow_MA'], line=dict(color='#d29922', width=1.5), name=f'Slow MA ({slow_period})'), row=1, col=1)
        
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Strategy'], line=dict(color='#3fb950', width=2), name="Strategy Equity"), row=2, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Market'], line=dict(color='#8b949e', width=1, dash='dash'), name="Buy & Hold"), row=2, col=1)
        
        fig_bt.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.warning("⚠️ Market data unavailable currently. Please click 'Reload / Retry Connection' on the left sidebar.")

# -------------------------------------------------------------
# TAB 2: OPTIONS QUANT TERMINAL
# -------------------------------------------------------------
with view_mode[1]:
    st.markdown(f"""
    <div class="hero-banner">
        <h2 style="margin:0; color:#f0f6fc;">Options Algorithmic Selection Engine</h2>
        <p style="margin:5px 0 0 0; color:#8b949e;">Optimal Strike Selection via Dynamic Delta & Volatility Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    ticker_obj, exp_dates = fetch_options_chain(symbol)
    
    if not exp_dates:
        st.warning("⚠️ No options chains available for this asset or market data feed is resting. Try re-clicking Retry.")
    else:
        o_col1, o_col2 = st.columns(2)
        selected_exp = o_col1.selectbox("Select Option Expiration Date:", exp_dates[:10])
        trade_dir = o_col2.radio("Market Bias:", ["CALL (Bullish) 📈", "PUT (Bearish) 📉"], horizontal=True)
        
        is_call_type = "CALL" in trade_dir
        
        opt_tp_stock = round(live_price + (1.5 * atr_val) if is_call_type else live_price - (1.5 * atr_val), 2)
        opt_sl_stock = round(live_price - (1.0 * atr_val) if is_call_type else live_price + (1.0 * atr_val), 2)
        
        st.info(f"💡 **Automated Volatility Targets:** Stock Target = **${opt_tp_stock}** | Stock Stop = **${opt_sl_stock}**")
        
        try:
            chain = ticker_obj.option_chain(selected_exp)
            opts_df = chain.calls if is_call_type else chain.puts
            
            opts = opts_df[(opts_df['strike'] >= live_price * 0.85) & (opts_df['strike'] <= live_price * 1.15)].copy()
            opts = opts[opts['ask'] > 0.05].copy()
            
            if not opts.empty:
                results = []
                dp = abs(opt_tp_stock - live_price)
                ds = abs(live_price - opt_sl_stock)
                
                for _, row in opts.iterrows():
                    strike = row['strike']
                    ask = row['ask']
                    bid = row['bid']
                    vol = row['volume'] if not np.isnan(row['volume']) else 0
                    
                    moneness = (live_price - strike) if is_call_type else (strike - live_price)
                    delta = min(0.85, max(0.15, 0.50 + (moneness / live_price) * 2.8))
                    
                    tp_price = ask + (dp * delta)
                    sl_price = max(0.01, ask - (ds * delta))
                    
                    profit = tp_price - ask
                    loss = ask - sl_price
                    rr = round(profit / loss, 2) if loss > 0 else 0
                    roi = (profit / ask) * 100
                    
                    score = (roi * 0.4) + (rr * 15) + (np.log1p(vol) * 2.0)
                    
                    results.append({
                        "Strike": strike, "Ask": ask, "Bid": bid, "Volume": int(vol),
                        "Delta": round(delta, 2), "Opt TP": round(tp_price, 2),
                        "Opt SL": round(sl_price, 2), "ROI %": round(roi, 1),
                        "R:R": rr, "Score": score
                    })
                
                res_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
                top_opt = res_df.iloc[0]
                
                st.markdown(f"""
                <div class="hero-banner" style="border-color:#3fb950; background: rgba(63, 185, 80, 0.1);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:#238636; color:#fff; padding:3px 10px; border-radius:8px; font-weight:bold; font-size:0.8rem;">RECOMMENDED CONTRACT</span>
                            <h1 style="margin:10px 0 0 0; color:#f0f6fc;">{symbol} ${top_opt['Strike']:.1f} {'CALL' if is_call_type else 'PUT'}</h1>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:#8b949e; font-size:0.9rem;">Ask Price</span>
                            <h2 style="margin:0; color:#58a6ff;">${top_opt['Ask']:.2f}</h2>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                o1, o2, o3, o4 = st.columns(4)
                o1.markdown(f'<div class="metric-card"><div class="metric-label">Option Target (TP)</div><div class="metric-val-green">${top_opt["Opt TP"]:.2f} (+{top_opt["ROI %"]}%)</div></div>', unsafe_allow_html=True)
                o2.markdown(f'<div class="metric-card"><div class="metric-label">Option Stop Loss (SL)</div><div class="metric-val-red">${top_opt["Opt SL"]:.2f}</div></div>', unsafe_allow_html=True)
                o3.markdown(f'<div class="metric-card"><div class="metric-label">Risk/Reward Ratio</div><div class="metric-val-blue">1:{top_opt["R:R"]}</div></div>', unsafe_allow_html=True)
                o4.markdown(f'<div class="metric-card"><div class="metric-label">Expected Net Profit</div><div class="metric-val-green">+${(top_opt["Opt TP"]-top_opt["Ask"])*100:.0f}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Ranked Options Chain Matrix")
                st.dataframe(res_df.drop(columns=['Score']), use_container_width=True, height=350)
                
            else:
                st.warning("No high-liquidity options contracts available for this criteria.")
        except Exception:
            st.error("Error evaluating options chain data.")
