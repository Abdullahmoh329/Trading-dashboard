import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
datetime_mod = __import__('datetime')

# -------------------------------------------------------------
# 1. Page Configuration & Finova Dark Theme CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Finova - Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #94a3b8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
        padding-top: 20px;
    }
    
    /* Finova Logo Header in Sidebar */
    .finova-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 16px 24px 16px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
    }
    .finova-logo-icon {
        background: #10b981;
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
    .finova-logo-text {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Cards & Metric Containers */
    .finova-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
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
    
    /* Badges & Icons in Cards */
    .card-icon-box {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        float: right;
    }
    
    /* Streamlit Radio / Navigation override */
    stRadio > label { font-weight: 600; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Resilient Data Engine with Smart Fallback
# -------------------------------------------------------------
def generate_mock_data(ticker_symbol):
    np.random.seed(hash(ticker_symbol) % 2035)
    base_prices = {"AMD": 165.0, "NVDA": 125.0, "TSLA": 220.0, "AAPL": 215.0}
    start_p = base_prices.get(ticker_symbol, 150.0)
    
    dates = pd.date_range(end=datetime_mod.date.today(), periods=120, freq='B')
    returns = np.random.normal(0.001, 0.022, len(dates))
    price_path = start_p * np.cumprod(1 + returns)
    
    df = pd.DataFrame(index=dates)
    df['Close'] = price_path
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.005, len(dates)))
    df['High'] = df[['Close', 'Open']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.008, len(dates))))
    df['Low'] = df[['Close', 'Open']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.008, len(dates))))
    df['Volume'] = np.random.randint(40000000, 150000000, len(dates))
    
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    curr_price = float(df['Close'].iloc[-1])
    
    return df, curr_price, atr

@st.cache_data(ttl=60)
def fetch_stock_data(ticker_symbol, timeframe="6m"):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=timeframe)
        if df is None or df.empty:
            df = yf.download(ticker_symbol, period=timeframe, progress=False)
            
        if df is not None and not df.empty:
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
        pass
    return generate_mock_data(ticker_symbol)

def get_options_data(ticker_symbol, live_price):
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = list(ticker.options)
        if expirations:
            return ticker, expirations
    except Exception:
        pass
    
    class MockOptionChain:
        def __init__(self, lp):
            strikes = np.linspace(round(lp * 0.85, 0), round(lp * 1.15, 0), 15)
            self.calls = pd.DataFrame({
                'strike': strikes,
                'ask': np.round(np.maximum(0.5, (lp - strikes) * 0.1 + np.random.uniform(2, 8, len(strikes))), 2),
                'bid': np.round(np.maximum(0.2, (lp - strikes) * 0.1 + np.random.uniform(1, 7, len(strikes))), 2),
                'volume': np.random.randint(500, 15000, len(strikes))
            })
            self.puts = pd.DataFrame({
                'strike': strikes,
                'ask': np.round(np.maximum(0.5, (strikes - lp) * 0.1 + np.random.uniform(2, 8, len(strikes))), 2),
                'bid': np.round(np.maximum(0.2, (strikes - lp) * 0.1 + np.random.uniform(1, 7, len(strikes))), 2),
                'volume': np.random.randint(500, 15000, len(strikes))
            })

    class MockTicker:
        def __init__(self, lp):
            self.lp = lp
        def option_chain(self, date):
            return MockOptionChain(self.lp)

    today = datetime_mod.date.today()
    mock_exp = [(today + datetime_mod.timedelta(days=i)).strftime('%Y-%m-%d') for i in [7, 14, 21, 35, 49]]
    return MockTicker(live_price), mock_exp

# -------------------------------------------------------------
# 3. Strategy & Pattern Logic
# -------------------------------------------------------------
def detect_patterns(df):
    patterns = []
    if len(df) < 30:
        return ["Insufficient data history for pattern detection"]
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    r_highs = highs[-20:]
    r_lows = lows[-20:]

    min1_idx = np.argmin(r_lows[:10])
    min2_idx = np.argmin(r_lows[10:]) + 10
    if abs(r_lows[min1_idx] - r_lows[min2_idx]) / r_lows[min1_idx] < 0.025:
        patterns.append("🟢 Potential Double Bottom (Bullish Reversal)")

    max1_idx = np.argmax(r_highs[:10])
    max2_idx = np.argmax(r_highs[10:]) + 10
    if abs(r_highs[max1_idx] - r_highs[max2_idx]) / r_highs[max1_idx] < 0.025:
        patterns.append("🔴 Potential Double Top (Bearish Reversal)")

    initial_move = (closes[-15] - closes[-30]) / closes[-30]
    recent_range = (max(r_highs[-10:]) - min(r_lows[-10:])) / min(r_lows[-10:])
    if initial_move > 0.03 and recent_range < 0.04:
        patterns.append("🚀 Bull Flag Consolidation")
    elif initial_move < -0.03 and recent_range < 0.04:
        patterns.append("⚠️ Bear Flag Consolidation")

    if not patterns:
        patterns.append("➡️ Market in Standard Consolidation (No Classic Pattern Triggered)")
    return patterns

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
        "net_profit": net_profit, "ret_pct": ret_pct,
        "win_rate": win_rate, "max_dd": max_drawdown, "trades": total_trades
    }

# -------------------------------------------------------------
# 4. Finova Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.markdown("""
<div class="finova-logo">
    <div class="finova-logo-icon">F</div>
    <div class="finova-logo-text">Finova</div>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📈 Stock Analytics & Backtest", "🎯 Options Quant Selector", "💼 Investments & Accounts", "⚙️ Insights & Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("Active Ticker Symbol:", value="AMD").upper().strip()
stock_df, live_price, atr_val = fetch_stock_data(symbol)

st.sidebar.markdown(f"**Current Price:** `${live_price:.2f}`")
st.sidebar.markdown(f"**14-ATR Volatility:** `${atr_val:.2f}`")

if st.sidebar.button("🔄 Refresh Data Feed"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Sign Out"):
    st.info("Signed out successfully.")

# Current Date formatting matching Finova reference
current_date_str = datetime_mod.date.today().strftime("%A, %B %d, %Y")

# -------------------------------------------------------------
# VIEW 1: FINOVA DASHBOARD
# -------------------------------------------------------------
if nav_selection == "📊 Dashboard":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">Dashboard</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">{current_date_str}</p>
    """, unsafe_allow_html=True)
    
    # Top 4 Metric Cards matching Finova layout
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="finova-card">
            <div class="card-icon-box" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">💳</div>
            <div class="card-label">Net Worth</div>
            <div class="card-value">$56,225</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="finova-card">
            <div class="card-icon-box" style="background: rgba(59, 130, 246, 0.15); color: #3b82f6;">📉</div>
            <div class="card-label">Monthly Income</div>
            <div class="card-value">$7,500</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="finova-card">
            <div class="card-icon-box" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b;">📈</div>
            <div class="card-label">Monthly Expenses</div>
            <div class="card-value">$1,828</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="finova-card">
            <div class="card-icon-box" style="background: rgba(139, 92, 246, 0.15); color: #8b5cf6;">📊</div>
            <div class="card-label">Investments</div>
            <div class="card-value">$25,285</div>
            <div style="font-size: 0.75rem; color: #10b981; margin-top: 6px; font-weight: 600;">↗ +28.1% total return</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Section matching Finova layout
    ch1, ch2 = st.columns([2, 1])
    with ch1:
        st.markdown("""
        <div class="finova-card">
            <h3 style="color: #f8fafc; font-size: 1.1rem; margin-top:0; margin-bottom: 15px;">Income vs Expenses</h3>
        """, unsafe_allow_html=True)
        
        # Mock chart matching Finova visual
        fig_inc = go.Figure()
        months = ['Feb', 'Mar', 'Apr', 'May', 'Jun']
        inc_vals = [7500, 7500, 7500, 7500, 8800]
        exp_vals = [1800, 1800, 2100, 1600, 2400]
        
        fig_inc.add_trace(go.Scatter(x=months, y=inc_vals, mode='lines', line=dict(color='#10b981', width=3), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)', name='Income'))
        fig_inc.add_trace(go.Scatter(x=months, y=exp_vals, mode='lines', line=dict(color='#f43f5e', width=3), fill='tozeroy', fillcolor='rgba(244, 63, 94, 0.1)', name='Expenses'))
        fig_inc.update_layout(template="plotly_dark", height=280, paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with ch2:
        st.markdown("""
        <div class="finova-card">
            <h3 style="color: #f8fafc; font-size: 1.1rem; margin-top:0; margin-bottom: 15px;">Expense Breakdown</h3>
        """, unsafe_allow_html=True)
        
        fig_pie = go.Figure(data=[go.Pie(labels=['Trading & Options', 'Gym & Fitness', 'Software', 'Others'], values=[55, 20, 15, 10], hole=.7, marker_colors=['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b'])])
        fig_pie.update_layout(template="plotly_dark", height=280, paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# VIEW 2: STOCK ANALYTICS & BACKTESTING
# -------------------------------------------------------------
elif nav_selection == "📈 Stock Analytics & Backtest":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">Stock Analytics & Backtesting: <span style="color:#3b82f6;">{symbol}</span></h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">Algorithmic Pattern Recognition & Moving Average Strategy</p>
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
            <div class="finova-card">
                <div class="card-label">Pattern Model Status</div>
                <div style="color: #3b82f6; font-size: 1.2rem; font-weight: bold;">Active Analysis</div>
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
        m1.markdown(f'<div class="finova-card"><div class="card-label">Net Profit</div><div class="{"card-value"}" style="color: {"#10b981" if metrics["net_profit"]>=0 else "#f43f5e"};">${metrics["net_profit"]:.2f} ({metrics["ret_pct"]:.1f}%)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="finova-card"><div class="card-label">Win Rate</div><div class="card-value" style="color: #3b82f6;">{metrics["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="finova-card"><div class="card-label">Max Drawdown</div><div class="card-value" style="color: #f43f5e;">{metrics["max_dd"]:.1f}%</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="finova-card"><div class="card-label">Total Trades</div><div class="card-value" style="color: #3b82f6;">{metrics["trades"]}</div></div>', unsafe_allow_html=True)
        
        fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        fig_bt.add_trace(go.Candlestick(x=bt_df.index, open=bt_df['Open'], high=bt_df['High'], low=bt_df['Low'], close=bt_df['Close'], name="Stock Price"), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Fast_MA'], line=dict(color='#3b82f6', width=1.5), name=f'Fast MA'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Slow_MA'], line=dict(color='#f59e0b', width=1.5), name=f'Slow MA'), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Strategy'], line=dict(color='#10b981', width=2), name="Strategy Equity"), row=2, col=1)
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Market'], line=dict(color='#94a3b8', width=1, dash='dash'), name="Buy & Hold"), row=2, col=1)
        
        fig_bt.update_layout(template="plotly_dark", height=500, paper_bgcolor="#111827", plot_bgcolor="#111827", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_bt, use_container_width=True)

# -------------------------------------------------------------
# VIEW 3: OPTIONS QUANT SELECTOR
# -------------------------------------------------------------
elif nav_selection == "🎯 Options Quant Selector":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">Options Algorithmic Selection Engine</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">Optimal Strike Selection via Dynamic Delta & Volatility Metrics</p>
    """, unsafe_allow_html=True)
    
    ticker_obj, exp_dates = get_options_data(symbol, live_price)
    
    if not exp_dates:
        st.warning("⚠️ No options chains available.")
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
                <div class="finova-card" style="border-color:#10b981; background: rgba(16, 185, 129, 0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:#10b981; color:#fff; padding:3px 10px; border-radius:6px; font-weight:bold; font-size:0.75rem;">RECOMMENDED CONTRACT</span>
                            <h2 style="margin:10px 0 0 0; color:#f8fafc;">{symbol} ${top_opt['Strike']:.1f} {'CALL' if is_call_type else 'PUT'}</h2>
                        </div>
                        <div style="text-align:right;">
                            <span style="color:#94a3b8; font-size:0.85rem;">Ask Price</span>
                            <h2 style="margin:0; color:#3b82f6;">${top_opt['Ask']:.2f}</h2>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                o1, o2, o3, o4 = st.columns(4)
                o1.markdown(f'<div class="finova-card"><div class="card-label">Option Target (TP)</div><div class="card-value" style="color:#10b981;">${top_opt["Opt TP"]:.2f} (+{top_opt["ROI %"]}%)</div></div>', unsafe_allow_html=True)
                o2.markdown(f'<div class="finova-card"><div class="card-label">Option Stop Loss (SL)</div><div class="card-value" style="color:#f43f5e;">${top_opt["Opt SL"]:.2f}</div></div>', unsafe_allow_html=True)
                o3.markdown(f'<div class="finova-card"><div class="card-label">Risk/Reward Ratio</div><div class="card-value" style="color:#3b82f6;">1:{top_opt["R:R"]}</div></div>', unsafe_allow_html=True)
                o4.markdown(f'<div class="finova-card"><div class="card-label">Expected Net Profit</div><div class="card-value" style="color:#10b981;">+${(top_opt["Opt TP"]-top_opt["Ask"])*100:.0f}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Ranked Options Chain Matrix")
                st.dataframe(res_df.drop(columns=['Score']), use_container_width=True, height=350)
        except Exception:
            st.error("Error evaluating options chain data.")

# -------------------------------------------------------------
# VIEW 4: INVESTMENTS & ACCOUNTS
# -------------------------------------------------------------
elif nav_selection == "💼 Investments & Accounts":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">Investments & Accounts</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">Portfolio Breakdown & Active Assets</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="finova-card">
        <h3>Active Brokerage Accounts</h3>
        <p>• <b>Primary Options Account:</b> $25,285 (Active)</p>
        <p>• <b>Long-term Equities:</b> $30,940</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# VIEW 5: INSIGHTS & SETTINGS
# -------------------------------------------------------------
elif nav_selection == "⚙️ Insights & Settings":
    st.markdown(f"""
        <h1 style="color: #f8fafc; font-weight: 800; margin-bottom: 0px;">Insights & Settings</h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px; margin-bottom: 24px;">System Preferences & Quant Configuration</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="finova-card">
        <h3>System Configuration</h3>
        <p>• Theme: Finova Dark Mode Pro</p>
        <p>• Data Engine: Resilient Fallback Enabled</p>
        <p>• Notification Alerts: Active</p>
    </div>
    """, unsafe_allow_html=True)
