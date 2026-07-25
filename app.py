import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. تهيئة الصفحة والتصميم
# -------------------------------------------------------------
st.set_page_config(page_title="Live Options Finder Terminal", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .best-card {
        background: linear-gradient(135deg, #1f242d 0%, #161b22 100%);
        border: 2px solid #58a6ff; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px; text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; }
    .metric-val { font-size: 1.25rem; font-weight: bold; color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. القائمة الجانبية وإدخال الهدف ووقف الخسارة
# -------------------------------------------------------------
st.sidebar.title("🎯 Live Options Selector")
st.sidebar.caption("محدد أفضل عقد أوبشن بناءً على هدف ووقف السهم")

symbol = st.sidebar.text_input("رمز السهم:", value="NVDA").upper().strip()

# جلب بيانات السهم المباشرة
@st.cache_data(ttl=30)
def get_stock_data(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period="5d", interval="15m")
    if hist.empty:
        return None, None, []
    current_price = float(hist['Close'].iloc[-1])
    expirations = t.options
    return t, current_price, expirations

ticker_obj, live_price, all_expirations = get_stock_data(symbol)

if not live_price:
    st.error(f"⚠️ يتعذر جلب بيانات السهم {symbol}. تأكد من الرمز.")
else:
    st.sidebar.markdown(f"**السعر الحالي للسهم:** `${live_price:.2f}`")
    
    # اختيار الاتجاه والهدف ووقف الخسارة
    trade_type = st.sidebar.radio("نوع الصفقة المستهدفة:", ["CALL (صاعد) 📈", "PUT (هابط) 📉"])
    
    default_tp = round(live_price * 1.02 if "CALL" in trade_type else live_price * 0.98, 2)
    default_sl = round(live_price * 0.99 if "CALL" in trade_type else live_price * 1.01, 2)

    target_price = st.sidebar.number_input("هدف السهم (Take Profit $):", value=default_tp, step=0.5)
    stop_loss = st.sidebar.number_input("وقف خسارة السهم (Stop Loss $):", value=default_sl, step=0.5)
    
    # اختيار تاريخ الانتهاء المستهدف (أو اقرب تاريخ)
    selected_exp = st.sidebar.selectbox("تاريخ انتهاء العقد (Expiration):", all_expirations[:5] if all_expirations else ["لا يوجد"])

    # -------------------------------------------------------------
    # 3. محرك فحص وتقييم عقود الأوبشن الحية
    # -------------------------------------------------------------
    def analyze_live_options(ticker, exp_date, current_s, tp_s, sl_s, is_call):
        try:
            chain = ticker.option_chain(exp_date)
            df_opts = chain.calls if is_call else chain.puts
            
            if df_opts.empty:
                return None

            # فلترة العقود القريبة من السعر (±10% من سعر السهم) والابتعاد عن العقود البعيدة جداً
            df_opts = df_opts[(df_opts['strike'] >= current_s * 0.85) & (df_opts['strike'] <= current_s * 1.15)].copy()
            
            # فلترة العقود ذات السيولة الضعيفة
            df_opts = df_opts[df_opts['ask'] > 0.05].copy()

            results = []
            
            for idx, row in df_opts.iterrows():
                strike = row['strike']
                ask_price = row['ask'] # سعر شراء العقد الحالي
                bid_price = row['bid']
                volume = row['volume'] if not np.isnan(row['volume']) else 0
                open_interest = row['openInterest'] if not np.isnan(row['openInterest']) else 0

                if ask_price <= 0:
                    continue

                # حساب التغير المتوقع في سعر السهم عند الهدف وعند الوقف
                price_diff_tp = abs(tp_s - current_s)
                price_diff_sl = abs(current_s - sl_s)

                # تقدير الدلتا التخمينية بناءً على القرب من السترايك (In The Money / Out Of The Money)
                moneness = (current_s - strike) if is_call else (strike - current_s)
                
                # تقدير تقريبي للدلتا (Delta) لتحديد استجابة العقد
                if moneness > 0: # ITM
                    est_delta = min(0.85, 0.50 + (moneness / current_s) * 3)
                else: # OTM
                    est_delta = max(0.15, 0.50 + (moneness / current_s) * 3)

                # ربح العقد عند هدف السهم (تقديري)
                est_opt_tp = ask_price + (price_diff_tp * est_delta)
                opt_profit = est_opt_tp - ask_price
                opt_roi = (opt_profit / ask_price) * 100

                # خسارة العقد عند وقف خسارة السهم (تقديري)
                est_opt_sl = max(0.01, ask_price - (price_diff_sl * est_delta))
                opt_loss = ask_price - est_opt_sl
                opt_risk_pct = (opt_loss / ask_price) * 100

                # نسبة العائد للمخاطرة للعقد (Risk/Reward Ratio)
                rr_ratio = round(opt_profit / opt_loss, 2) if opt_loss > 0 else 0

                results.append({
                    "strike": strike,
                    "ask": ask_price,
                    "bid": bid_price,
                    "volume": int(volume),
                    "openInterest": int(open_interest),
                    "est_delta": round(est_delta, 2),
                    "opt_tp_price": round(est_opt_tp, 2),
                    "opt_sl_price": round(est_opt_sl, 2),
                    "opt_roi": round(opt_roi, 1),
                    "opt_risk_pct": round(opt_risk_pct, 1),
                    "rr_ratio": rr_ratio,
                    "score": (opt_roi * 0.4) + (rr_ratio * 20) + (np.log1p(volume) * 2) # معادلة تفضيل السيولة والربحية
                })

            res_df = pd.DataFrame(results)
            if not res_df.empty:
                res_df = res_df.sort_values(by="score", ascending=False)
            return res_df

        except Exception as e:
            st.error(f"خطأ أثناء جلب سلسلة العقود: {str(e)}")
            return None

    # -------------------------------------------------------------
    # 4. عرض النتائج والعقد الأفضل
    # -------------------------------------------------------------
    is_call_trade = "CALL" in trade_type
    
    st.title(f"⚡ تحليل عقود الأوبشن الحية: {symbol}")
    st.caption(f"تاريخ الانتهاء المحدد: **{selected_exp}** | سعر السهم اللحظي: **${live_price:.2f}**")

    opts_df = analyze_live_options(ticker_obj, selected_exp, live_price, target_price, stop_loss, is_call_trade)

    if opts_df is not None and not opts_df.empty:
        best_opt = opts_df.iloc[0] # العقد ذو النتيجة الأعلى

        # كارت العقد الموصى به
        st.markdown(f"""
        <div class="best-card">
            <h3 style="margin:0; color:#58a6ff;">🏆 العقد الأفضل الموصى به (Best Choice)</h3>
            <div style="font-size:1.6rem; font-weight:bold; margin:10px 0; color:#f0f6fc;">
                {symbol} ${best_opt['strike']} {'CALL' if is_call_trade else 'PUT'} — Exp: {selected_exp}
            </div>
            <p style="margin:0; color:#8b949e;">
                سعر الشراء الحالي (Ask): <b style="color:#f0f6fc;">${best_opt['ask']:.2f}</b> (${best_opt['ask']*100:.0f} للعقد) | 
                السيولة (Volume): <b style="color:#f0f6fc;">{best_opt['volume']}</b> | 
                الدلتا التقريبية: <b style="color:#f0f6fc;">{best_opt['est_delta']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # تفاصيل أهداف العقد المختار
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">هدف العقد عند TP السهم</div><div class="metric-val" style="color:#2ea043;">${best_opt["opt_tp_price"]:.2f} (+{best_opt["opt_roi"]}%)</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">وقف العقد عند SL السهم</div><div class="metric-val" style="color:#da3633;">${best_opt["opt_sl_price"]:.2f} (-{best_opt["opt_risk_pct"]}%)</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">نسبة العائد / المخاطرة</div><div class="metric-val" style="color:#58a6ff;">1:{best_opt["rr_ratio"]}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">الربح الصافي المتوقع / عقد</div><div class="metric-val" style="color:#2ea043;">+${(best_opt["opt_tp_price"] - best_opt["ask"])*100:.0f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 قائمة باقي السترايكات المتاحة والمفاضلة بينها:")

        # عرض الجدول كاملاً للمقارنة
        display_df = opts_df[['strike', 'ask', 'bid', 'est_delta', 'opt_tp_price', 'opt_roi', 'opt_risk_pct', 'rr_ratio', 'volume']].copy()
        display_df.columns = ['السترايك (Strike)', 'سعر الشراء (Ask)', 'سعر البيع (Bid)', 'Delta', 'هدف العقد ($)', 'نسبة الربح المتوقعة (%)', 'مخاطرة الوقف (%)', 'العائد/المخاطرة', 'حجم التداول (Volume)']
        
        st.dataframe(display_df, use_container_width=True)

    else:
        st.warning("لم يتم العثور على عقود تلبّي شروط السيولة في هذا التاريخ. اختر تاريخ انتهاء آخر من الشريط الجانبي.")
