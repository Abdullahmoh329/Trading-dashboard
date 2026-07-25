import yfinance as yf
import pandas as pd
import numpy as np

def get_quant_recommendation(symbol: str, timeframe: str = "15m"):
    """
    محرك كمي لتوليد توصيات السكالبينغ والأوبشنز بناءً على:
    1. VWAP اللحظي
    2. إعادة الاختبار (Retest)
    3. انفجار الفوليوم (Volume Spike)
    4. متوسطات الحركة (EMA 9/21/50)
    5. نطاق التذبذب المباشر (ATR)
    """
    try:
        # 1. جلب البيانات المباشرة (فريم 15د أو 5د)
        df = yf.download(symbol, period="5d", interval=timeframe, progress=False)
        
        if df.empty:
            return {"status": "error", "message": f"تعذر جلب بيانات السهم {symbol}"}
            
        # معالجة أعمدة yfinance المتعددة
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        if len(df) < 30:
            return {"status": "error", "message": "عدد الشموع غير كافٍ للتحليل"}

        # 2. الحسابات الفنية الحقيقية
        # أ) الـ VWAP اليومي
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['TP'] * df['Volume']
        df['Date'] = df.index.date
        df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
        df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
        df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']

        # ب) المتوسطات المتحركة
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # ج) الفوليوم و ATR
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA']
        
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # د) الدعوم والمقاومات للاختراق وإعادة الاختبار
        df['Res'] = df['High'].rolling(20).max().shift(1)
        df['Sup'] = df['Low'].rolling(20).min().shift(1)

        # 3. تحليل أحدث شمعة في السوق
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        curr_price = float(latest['Close'])
        vwap_val = float(latest['VWAP'])
        atr_val = float(latest['ATR']) if not np.isnan(latest['ATR']) else curr_price * 0.01

        score_call = 0
        score_put = 0
        signals_log = []

        # -- فحص الـ VWAP (25 نقطة) --
        if curr_price > vwap_val:
            score_call += 25
            signals_log.append("السعر يتداول أعلى من الـ VWAP (سيطرة شرائية).")
        else:
            score_put += 25
            signals_log.append("السعر يتداول أسفل الـ VWAP (ضغط بيعي).")

        # -- فحص ترتيب المتوسطات (25 نقطة) --
        if latest['EMA_9'] > latest['EMA_21'] > latest['EMA_50']:
            score_call += 25
            signals_log.append("اتجاه صاعد نقي (EMA 9 > 21 > 50).")
        elif latest['EMA_9'] < latest['EMA_21'] < latest['EMA_50']:
            score_put += 25
            signals_log.append("اتجاه هابط نقي (EMA 9 < 21 < 50).")

        # -- فحص الانفجار السيولي (25 نقطة) --
        if latest['Vol_Ratio'] >= 1.3:
            if curr_price > latest['Open']:
                score_call += 25
                signals_log.append(f"سيولة شرائية عالية ({latest['Vol_Ratio']:.1f}x ضعف المتوسط).")
            else:
                score_put += 25
                signals_log.append(f"سيولة بيعية عالية ({latest['Vol_Ratio']:.1f}x ضعف المتوسط).")

        # -- فحص إعادة الاختبار Retest (25 نقطة) --
        # إعادة اختبار المقاومة كدعم (Call)
        if prev['Close'] > prev['Res'] and latest['Low'] <= latest['Res'] * 1.001 and curr_price >= latest['Res']:
            score_call += 25
            signals_log.append("🎯 إعادة اختبار (Retest) ناجحة للمقاومة المكسورة.")
        # إعادة اختبار الدعم كمقاومة (Put)
        elif prev['Close'] < prev['Sup'] and latest['High'] >= latest['Sup'] * 0.999 and curr_price <= latest['Sup']:
            score_put += 25
            signals_log.append("🎯 إعادة اختبار (Retest) ناجحة للدعم المكسور.")

        # 4. بناء التوصية الحسابية
        if score_call >= 60:
            action = "STRONG CALL"
            color = "green"
            sl = round(curr_price - (1.2 * atr_val), 2)
            tp1 = round(curr_price + (1.5 * atr_val), 2)
            tp2 = round(curr_price + (2.5 * atr_val), 2)
            confidence = score_call
        elif score_put >= 60:
            action = "STRONG PUT"
            color = "red"
            sl = round(curr_price + (1.2 * atr_val), 2)
            tp1 = round(curr_price - (1.5 * atr_val), 2)
            tp2 = round(curr_price - (2.5 * atr_val), 2)
            confidence = score_put
        else:
            action = "NEUTRAL / WAIT"
            color = "gray"
            sl = round(curr_price * 0.99, 2)
            tp1 = round(curr_price * 1.01, 2)
            tp2 = round(curr_price * 1.02, 2)
            confidence = max(score_call, score_put)

        # 5. إرجاع النتيجة كـ Dictionary نظيف يستقبله موقعك
        return {
            "status": "success",
            "symbol": symbol,
            "current_price": round(curr_price, 2),
            "vwap": round(vwap_val, 2),
            "action": action,
            "color": color,
            "confidence": confidence,
            "entry": round(curr_price, 2),
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": tp2,
            "reasons": signals_log
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
