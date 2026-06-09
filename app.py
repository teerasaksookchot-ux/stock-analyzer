import streamlit as st
import yfinance as yf
import anthropic
import requests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

st.set_page_config(page_title="Stock Analyzer", layout="wide")
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ===== SESSION CACHE =====
if "history" not in st.session_state:
    st.session_state["history"] = {}
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = {}

# ===== PARSE UPLOADED FILE =====
def parse_analysis_file(text: str) -> dict | None:
    try:
        markers = {
            "[Financial Analysis]":    "fin_result",
            "[Macro Analysis]":        "mac_result",
            "[Geopolitical Analysis]": "geo_result",
            "[News Analysis]":         "news_result",
            "[Technical Analysis]":    "tech_result",
            "[CIO Full Report]":       "final",
            "[Chat History]":          "chat_raw",
        }
        sections = {v: "" for v in markers.values()}
        first_line = text.strip().split("\n")[0]
        parts  = first_line.replace("===", "").strip().split("—")
        ticker = parts[0].strip()
        name   = parts[-1].strip() if len(parts) > 1 else "N/A"
        current = None
        for line in text.split("\n"):
            if line.strip() in markers:
                current = markers[line.strip()]
                continue
            if current:
                sections[current] += line + "\n"
        for k in sections:
            sections[k] = sections[k].strip()

        # แปลง Chat History กลับเป็น list of dicts
        chat_messages = []
        if sections["chat_raw"]:
            current_role    = None
            current_content = []
            for line in sections["chat_raw"].split("\n"):
                if line.startswith("User: "):
                    if current_role:
                        chat_messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role    = "user"
                    current_content = [line[6:]]
                elif line.startswith("Agent: "):
                    if current_role:
                        chat_messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role    = "assistant"
                    current_content = [line[7:]]
                elif current_role:
                    current_content.append(line)
            if current_role and current_content:
                chat_messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

        company = {
            "ticker": ticker, "name": name,
            "price": 0, "market_cap_b": 0,
            "sector": "N/A", "industry": "N/A", "summary": "",
        }
        return {
            "company":      company,
            "fin_result":   sections["fin_result"],
            "mac_result":   sections["mac_result"],
            "geo_result":   sections["geo_result"],
            "news_result":  sections["news_result"],
            "tech_result":  sections["tech_result"],
            "final":        sections["final"],
            "chat_messages": chat_messages,
        }
    except:
        return None

# ===== SIDEBAR =====
with st.sidebar:
    st.subheader("ประวัติการวิเคราะห์")
    if st.session_state["history"]:
        for saved_ticker in list(st.session_state["history"].keys()):
            col1, col2 = st.sidebar.columns([3, 1])
            if col1.button(saved_ticker, key=f"load_{saved_ticker}"):
                st.session_state["load_ticker"] = saved_ticker
            if col2.button("✕", key=f"del_{saved_ticker}"):
                del st.session_state["history"][saved_ticker]
                st.session_state["chat_messages"].pop(saved_ticker, None)
                st.rerun()
    else:
        st.caption("ยังไม่มีประวัติในเซสชันนี้")

    st.divider()

    st.subheader("โหลดไฟล์เก่า")
    uploaded = st.file_uploader(
        "อัปโหลดไฟล์ .txt ที่เคย Download ไว้",
        type=["txt"],
        help="ไฟล์จากปุ่ม Download ใน CIO Full Report"
    )
    if uploaded:
        text   = uploaded.read().decode("utf-8")
        parsed = parse_analysis_file(text)
        if parsed:
            t = parsed["company"]["ticker"]
            st.session_state["history"][t] = parsed
            if parsed.get("chat_messages"):
                st.session_state["chat_messages"][t] = parsed["chat_messages"]
                st.success(f"โหลด {t} สำเร็จ (มีประวัติแชท {len(parsed['chat_messages'])} ข้อความ)")
            else:
                st.success(f"โหลด {t} สำเร็จ")
            st.session_state["load_ticker"] = t
            st.rerun()
        else:
            st.error("ไฟล์ไม่ถูกต้อง กรุณาใช้ไฟล์จากปุ่ม Download เท่านั้น")

    st.divider()
    st.caption("ปิด browser = ประวัติหาย\nกด Download เพื่อเก็บไฟล์ไว้")

# ===== DATA NODES =====

def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        if not info.get("currentPrice") and not info.get("regularMarketPrice"):
            return None
        return {
            "ticker":       ticker,
            "name":         info.get("shortName", "N/A"),
            "price":        info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1),
            "sector":       info.get("sector", "N/A"),
            "industry":     info.get("industry", "N/A"),
            "summary":      info.get("longBusinessSummary", "N/A")[:500],
        }
    except:
        return None

def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        def df_to_str(df, name):
            if df is None or df.empty:
                return f"{name}: ไม่มีข้อมูล"
            return f"{name}:\n{df.to_string()}"
        return {
            "income_stmt":   df_to_str(stock.financials,    "Income Statement"),
            "balance_sheet": df_to_str(stock.balance_sheet, "Balance Sheet"),
            "cash_flow":     df_to_str(stock.cashflow,      "Cash Flow"),
        }
    except:
        return {"income_stmt": "N/A", "balance_sheet": "N/A", "cash_flow": "N/A"}

@st.cache_data(ttl=3600)
def get_chart_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="6mo"), stock.financials, stock.balance_sheet, stock.cashflow

def get_price_summary(hist):
    if hist is None or hist.empty:
        return "ไม่มีข้อมูลราคา"
    return (f"ราคาล่าสุด: ${hist['Close'].iloc[-1]:.2f} | "
            f"6M High: ${hist['Close'].max():.2f} | "
            f"6M Low: ${hist['Close'].min():.2f} | "
            f"MA20: ${hist['Close'].tail(20).mean():.2f} | "
            f"MA50: ${hist['Close'].tail(50).mean():.2f} | "
            f"Avg Volume: {int(hist['Volume'].mean()):,}")

def get_realtime_price(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("currentPrice") or info.get("regularMarketPrice", 0)
    except:
        return 0

def get_news(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "ไม่มีข่าว"
        return "\n".join([f"- {n.get('content', {}).get('title', 'N/A')}" for n in news[:10]])
    except:
        return "ไม่มีข่าว"

def get_macro_data():
    try:
        fed = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS", timeout=5).text.strip().split("\n")[-1]
        cpi = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", timeout=5).text.strip().split("\n")[-1]
        fd, fr = fed.split(",")
        cd, cv = cpi.split(",")
        return f"Fed Rate: {fr}% (ณ {fd}) | CPI: {cv} (ณ {cd})"
    except:
        return "ไม่สามารถดึงข้อมูล macro ได้"


def get_technical_indicators(hist):
    """คำนวณ technical indicators ครบชุด"""
    try:
        c = hist["Close"]
        h = hist["High"]
        l = hist["Low"]

        # RSI
        d   = c.diff()
        rsi = 100 - (100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()))

        # MACD
        ema12    = c.ewm(span=12, adjust=False).mean()
        ema26    = c.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()
        hist_val = macd - signal

        # ATR
        tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Stochastic %K %D
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        stoch_k = ((c - low14) / (high14 - low14) * 100)
        stoch_d = stoch_k.rolling(3).mean()

        # Bollinger Band Width
        ma20  = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_width = ((ma20 + 2*std20) - (ma20 - 2*std20)) / ma20 * 100

        latest = {
            "RSI":        round(rsi.iloc[-1], 1),
            "MACD":       round(macd.iloc[-1], 3),
            "MACD_Signal":round(signal.iloc[-1], 3),
            "MACD_Hist":  round(hist_val.iloc[-1], 3),
            "ATR":        round(atr.iloc[-1], 2),
            "Stoch_K":    round(stoch_k.iloc[-1], 1),
            "Stoch_D":    round(stoch_d.iloc[-1], 1),
            "BB_Width":   round(bb_width.iloc[-1], 1),
        }

        summary = (
            f"RSI(14)={latest['RSI']} | "
            f"MACD={latest['MACD']} Signal={latest['MACD_Signal']} Hist={latest['MACD_Hist']} | "
            f"ATR(14)=${latest['ATR']} | "
            f"Stoch %K={latest['Stoch_K']} %D={latest['Stoch_D']} | "
            f"BB Width={latest['BB_Width']}%"
        )
        return summary
    except Exception as e:
        return f"คำนวณ indicators ไม่ได้: {e}"

def get_quarterly_financials(ticker):
    """ดึงงบการเงินรายไตรมาส"""
    try:
        stock = yf.Ticker(ticker)
        def df_to_str(df, name):
            if df is None or df.empty:
                return f"{name}: ไม่มีข้อมูล"
            return f"{name}:\n{df.to_string()}"
        return {
            "quarterly_income": df_to_str(stock.quarterly_financials,    "Quarterly Income Statement"),
            "quarterly_balance": df_to_str(stock.quarterly_balance_sheet, "Quarterly Balance Sheet"),
            "quarterly_cashflow": df_to_str(stock.quarterly_cashflow,     "Quarterly Cash Flow"),
        }
    except:
        return {"quarterly_income": "N/A", "quarterly_balance": "N/A", "quarterly_cashflow": "N/A"}

def get_analyst_ratings(ticker):
    """ดึงคำแนะนำจากนักวิเคราะห์"""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        recs  = stock.recommendations
        target_price = info.get("targetMeanPrice")
        target_high  = info.get("targetHighPrice")
        target_low   = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions", 0)
        rec_key      = info.get("recommendationKey", "N/A")
        rec_mean     = info.get("recommendationMean")

        summary = f"Consensus: {rec_key.upper()} (mean={rec_mean}) | จาก {num_analysts} นักวิเคราะห์"
        if target_price:
            summary += f" | Target Price: ${target_price:.2f} (Low=${target_low:.2f} High=${target_high:.2f})"

        if recs is not None and not recs.empty:
            latest_recs = recs.tail(5).to_string()
            summary += f"\n\nคำแนะนำล่าสุด:\n{latest_recs}"
        return summary
    except:
        return "ไม่มีข้อมูลนักวิเคราะห์"

def get_earnings_date(ticker):
    """ดึงวันประกาศผลประกอบการถัดไป"""
    try:
        stock    = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is None or calendar.empty:
            return "ไม่มีข้อมูลวันประกาศผล"
        earnings_dates = []
        if "Earnings Date" in calendar.index:
            dates = calendar.loc["Earnings Date"]
            if hasattr(dates, '__iter__'):
                earnings_dates = [str(d)[:10] for d in dates]
            else:
                earnings_dates = [str(dates)[:10]]
        eps_est = calendar.loc["EPS Estimate"].values[0] if "EPS Estimate" in calendar.index else "N/A"
        rev_est = calendar.loc["Revenue Estimate"].values[0] if "Revenue Estimate" in calendar.index else "N/A"
        date_str = " ถึง ".join(earnings_dates) if earnings_dates else "N/A"
        return f"Earnings Date: {date_str} | EPS Estimate: {eps_est} | Revenue Estimate: {rev_est}"
    except:
        return "ไม่มีข้อมูลวันประกาศผล"

def get_yield_curve():
    """ดึง 2Y/10Y Treasury yield spread จาก FRED"""
    try:
        y2  = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",  timeout=5).text.strip().split("\n")[-1]
        y10 = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", timeout=5).text.strip().split("\n")[-1]
        _, r2  = y2.split(",")
        _, r10 = y10.split(",")
        spread = round(float(r10) - float(r2), 2)
        status = "Inverted (เสี่ยง recession)" if spread < 0 else "Normal"
        return f"2Y Yield: {r2}% | 10Y Yield: {r10}% | Spread: {spread}% ({status})"
    except:
        return "ไม่สามารถดึงข้อมูล Yield Curve"


@st.cache_data(ttl=1800)
def get_geopolitical_indicators():
    """ดึงตัวชี้วัดความเสี่ยงภูมิรัฐศาสตร์จากตลาด"""
    indicators = {
        "VIX (ความกลัวตลาด)":   "^VIX",
        "Gold (safe haven $)":   "GC=F",
        "Oil WTI ($/barrel)":    "CL=F",
        "USD Index":             "DX-Y.NYB",
        "Defense ETF (ITA)":     "ITA",
    }
    results = {}
    for name, sym in indicators.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if hist.empty:
                continue
            cur    = hist["Close"].iloc[-1]
            prev   = hist["Close"].iloc[0]
            chg    = round((cur - prev) / prev * 100, 1)
            sign   = "+" if chg >= 0 else ""
            results[name] = f"{cur:.2f} ({sign}{chg}% 5d)"
        except:
            pass
    return results


@st.cache_data(ttl=3600)
def get_macro_timeseries():
    """ดึง Fed Rate และ CPI ย้อนหลัง 6 เดือนเป็น time series"""
    try:
        cutoff   = datetime.now() - timedelta(days=180)
        fed_text = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS", timeout=5).text
        cpi_text = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", timeout=5).text
        fed_df   = pd.read_csv(io.StringIO(fed_text))
        cpi_df   = pd.read_csv(io.StringIO(cpi_text))
        fed_df.columns = cpi_df.columns = ["date", "value"]
        for df in [fed_df, cpi_df]:
            df["date"]  = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        fed_df = fed_df[fed_df["date"] >= cutoff].dropna()
        cpi_df = cpi_df[cpi_df["date"] >= cutoff].dropna()
        return fed_df, cpi_df
    except:
        return None, None

SECTOR_PEERS = {
    "Technology":             ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC"],
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "TXN"],
    "Industrials":            ["GE", "HON", "MMM", "CAT", "BA", "LMT", "RTX"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "HAL", "BP", "SHEL"],
    "Healthcare":             ["JNJ", "PFE", "MRK", "ABT", "TMO", "UNH", "AMGN"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "Consumer Staples":       ["PG", "KO", "PEP", "WMT", "COST", "CL", "GIS"],
    "Real Estate":            ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "SPG"],
    "Utilities":              ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"],
    "Materials":              ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ"],
}

@st.cache_data(ttl=3600)
def get_competitors(ticker, sector):
    """ดึงข้อมูลคู่แข่งใน sector เดียวกัน"""
    peers   = [p for p in SECTOR_PEERS.get(sector, []) if p != ticker][:4]
    results = []
    for p in peers:
        try:
            info = yf.Ticker(p).info
            if not info.get("currentPrice") and not info.get("regularMarketPrice"):
                continue
            results.append({
                "ticker":       p,
                "name":         info.get("shortName", p)[:20],
                "price":        info.get("currentPrice") or info.get("regularMarketPrice", 0),
                "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1),
                "pe":           info.get("trailingPE"),
                "pb":           info.get("priceToBook"),
                "rev_growth":   info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
            })
        except:
            pass
    return results

# ===== CHART CALCULATIONS =====

def calc_bollinger(hist, window=20, k=2):
    ma  = hist["Close"].rolling(window).mean()
    std = hist["Close"].rolling(window).std()
    return ma, ma + k * std, ma - k * std

def calc_trendlines(hist):
    closes = hist["Close"].values
    dates  = hist.index
    hi_idx = argrelextrema(closes, np.greater, order=5)[0]
    lo_idx = argrelextrema(closes, np.less,    order=5)[0]

    def make_line(idx_arr, closes, dates):
        if len(idx_arr) < 2:
            return None
        x1, x2 = idx_arr[-2], idx_arr[-1]
        y1, y2  = closes[x1], closes[x2]
        slope   = (y2 - y1) / (x2 - x1)
        proj_y  = round(y2 + slope * (len(closes) - 1 - x2), 2)
        return {"x": [dates[x1], dates[x2], dates[-1]],
                "y": [round(y1,2), round(y2,2), proj_y],
                "slope": slope}

    return make_line(hi_idx, closes, dates), make_line(lo_idx, closes, dates)

# ===== CHART DRAW =====

def draw_chart(name, ticker, hist, fin, bs, cf, sector='Technology'):
    H = 300
    M = dict(l=0, r=0, t=40, b=0)

    if name == "ราคา + MA":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],                    name="ราคา", line=dict(color="#378ADD", width=2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(), name="MA20", line=dict(color="#EF9F27", dash="dash")))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="MA50", line=dict(color="#1D9E75", dash="dash")))
        fig.update_layout(title=f"{ticker} ราคา + MA", height=H, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Bollinger Bands":
        ma, upper, lower = calc_bollinger(hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=upper,         name="Upper", line=dict(color="#E24B4A", dash="dash", width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=lower,         name="Lower", line=dict(color="#1D9E75", dash="dash", width=1), fill="tonexty", fillcolor="rgba(29,158,117,0.06)"))
        fig.add_trace(go.Scatter(x=hist.index, y=ma,            name="MA20",  line=dict(color="#EF9F27", width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="ราคา",  line=dict(color="#378ADD", width=2)))
        fig.update_layout(title="Bollinger Bands (20,2)", height=H, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Trendline":
        resistance, support = calc_trendlines(hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="ราคา", line=dict(color="#378ADD", width=2)))
        if resistance:
            d = "ขาขึ้น" if resistance["slope"] > 0 else "ขาลง"
            fig.add_trace(go.Scatter(x=resistance["x"], y=resistance["y"], name=f"Resistance ({d})", line=dict(color="#E24B4A", dash="dot", width=1.5)))
        if support:
            d = "ขาขึ้น" if support["slope"] > 0 else "ขาลง"
            fig.add_trace(go.Scatter(x=support["x"], y=support["y"], name=f"Support ({d})", line=dict(color="#1D9E75", dash="dot", width=1.5)))
        fig.update_layout(title="Trendline (Auto)", height=H, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Volume":
        colors = ["#E24B4A" if c < o else "#1D9E75" for c, o in zip(hist["Close"], hist["Open"])]
        fig = go.Figure(go.Bar(x=hist.index, y=hist["Volume"], marker_color=colors))
        fig.update_layout(title="Volume", height=H, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "RSI":
        d   = hist["Close"].diff()
        rsi = 100 - (100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI", line=dict(color="#7F77DD", width=2)))
        fig.add_hrect(y0=70, y1=100, fillcolor="#E24B4A", opacity=0.07, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="#1D9E75", opacity=0.07, line_width=0)
        fig.add_hline(y=70, line_dash="dash", line_color="#E24B4A", annotation_text="70")
        fig.add_hline(y=30, line_dash="dash", line_color="#1D9E75", annotation_text="30")
        fig.update_layout(title="RSI (14)", height=H, yaxis_range=[0,100], margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Revenue":
        if fin is not None and "Total Revenue" in fin.index:
            rev   = fin.loc["Total Revenue"].dropna().sort_index() / 1e9
            years = [str(d.year) for d in rev.index]
            vals  = [round(v,2) for v in rev.values]
            fig   = go.Figure(go.Bar(x=years, y=vals, marker_color="#378ADD", text=[f"${v}B" for v in vals], textposition="outside"))
            fig.update_layout(title="Revenue (B$)", height=H, margin=M)
            st.plotly_chart(fig, use_container_width=True)

    elif name == "Gross Margin":
        if fin is not None and "Gross Profit" in fin.index and "Total Revenue" in fin.index:
            mg    = (fin.loc["Gross Profit"].dropna() / fin.loc["Total Revenue"].dropna() * 100).sort_index().round(1)
            years = [str(d.year) for d in mg.index]
            fig   = go.Figure(go.Bar(x=years, y=list(mg.values), marker_color="#1D9E75", text=[f"{v}%" for v in mg.values], textposition="outside"))
            fig.update_layout(title="Gross Margin (%)", height=H, margin=M)
            st.plotly_chart(fig, use_container_width=True)

    elif name == "Free Cash Flow":
        if cf is not None and "Free Cash Flow" in cf.index:
            fcf    = cf.loc["Free Cash Flow"].dropna().sort_index() / 1e6
            years  = [str(d.year) for d in fcf.index]
            vals   = [round(v,0) for v in fcf.values]
            colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in vals]
            fig    = go.Figure(go.Bar(x=years, y=vals, marker_color=colors, text=[f"${v}M" for v in vals], textposition="outside"))
            fig.add_hline(y=0, line_color="gray", line_width=0.5)
            fig.update_layout(title="Free Cash Flow (M$)", height=H, margin=M)
            st.plotly_chart(fig, use_container_width=True)

    elif name == "Debt vs Cash":
        if bs is not None and not bs.empty:
            yl, cl, dl = [], [], []
            for col in sorted(bs.columns)[:4]:
                cash = debt = 0
                for k in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                    if k in bs.index:
                        v = bs.loc[k, col]
                        if v and str(v) != "nan":
                            cash = float(v) / 1e9; break
                for k in ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]:
                    if k in bs.index:
                        v = bs.loc[k, col]
                        if v and str(v) != "nan":
                            debt = float(v) / 1e9; break
                yl.append(str(col.year)); cl.append(round(cash,2)); dl.append(round(debt,2))
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Cash", x=yl, y=cl, marker_color="#1D9E75", text=[f"${v}B" for v in cl], textposition="outside"))
            fig.add_trace(go.Bar(name="Debt", x=yl, y=dl, marker_color="#E24B4A", text=[f"${v}B" for v in dl], textposition="outside"))
            fig.update_layout(title="Debt vs Cash (B$)", barmode="group", height=H, margin=M)
            st.plotly_chart(fig, use_container_width=True)


    elif name == "Macro Overlay":
        fed_df, cpi_df = get_macro_timeseries()
        if fed_df is not None and not fed_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="ราคา",
                                      line=dict(color="#378ADD", width=2), yaxis="y1"))
            fig.add_trace(go.Scatter(x=fed_df["date"], y=fed_df["value"], name="Fed Rate (%)",
                                      line=dict(color="#E24B4A", dash="dash", width=1.5), yaxis="y2"))
            fig.update_layout(
                title="ราคาหุ้น vs Fed Rate",
                height=300, margin=dict(l=0, r=60, t=40, b=0),
                yaxis=dict(title="ราคา ($)"),
                yaxis2=dict(title="Fed Rate (%)", overlaying="y", side="right",
                             showgrid=False, tickformat=".2f"),
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่สามารถดึงข้อมูล Macro ได้")

    elif name == "Peer Comparison":
        competitors = get_competitors(ticker, sector)
        if competitors:
            peers_all = [{"ticker": ticker, "name": "This Stock",
                           "market_cap_b": 0, "pe": None, "gross_margin": None,
                           "rev_growth": None}] + competitors
            tickers   = [p["ticker"] for p in competitors]
            caps      = [p["market_cap_b"] for p in competitors]
            pes       = [p["pe"] if p["pe"] else 0 for p in competitors]
            margins   = [round(p["gross_margin"]*100, 1) if p["gross_margin"] else 0 for p in competitors]
            growths   = [round(p["rev_growth"]*100, 1) if p["rev_growth"] else 0 for p in competitors]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Market Cap (B$)", x=tickers, y=caps,   marker_color="#378ADD"))
            fig.add_trace(go.Bar(name="P/E",             x=tickers, y=pes,    marker_color="#EF9F27"))
            fig.add_trace(go.Bar(name="Gross Margin (%)", x=tickers, y=margins, marker_color="#1D9E75"))
            fig.update_layout(title="Peer Comparison", barmode="group", height=300,
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่พบข้อมูลคู่แข่งใน sector นี้")

# ===== AGENTS =====

def run_agent(prompt, max_tokens=1000):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

def financial_agent(company, financials, quarterly, analyst):
    return run_agent(f"""คุณเป็น Financial Analyst วิเคราะห์งบการเงินของ {company['ticker']}

=== งบการเงินรายปี ===
{financials['income_stmt']}
{financials['balance_sheet']}
{financials['cash_flow']}

=== งบการเงินรายไตรมาส ===
{quarterly['quarterly_income']}
{quarterly['quarterly_cashflow']}

=== คำแนะนำนักวิเคราะห์ ===
{analyst}

วิเคราะห์:
1. แนวโน้มรายได้และกำไร (รายปีและรายไตรมาส momentum เร่งหรือชะลอ)
2. ความแข็งแกร่งของงบดุล
3. คุณภาพกระแสเงินสด
4. จุดแข็ง/อ่อนจากงบ
5. consensus นักวิเคราะห์สอดคล้องกับงบมั้ย
ตอบเป็นภาษาไทย กระชับ อ้างอิงตัวเลขจริง""", 1200)

def macro_agent(company, macro, yield_curve):
    return run_agent(f"""คุณเป็น Macro Economist วิเคราะห์ผลกระทบ macro ต่อ {company['ticker']}
Macro: {macro}
Yield Curve: {yield_curve}
Sector: {company['sector']} | ธุรกิจ: {company['summary']}
วิเคราะห์:
1. ผลกระทบดอกเบี้ยและเงินเฟ้อต่อธุรกิจนี้
2. Yield Curve บ่งบอกอะไรและกระทบอย่างไร
3. macro เอื้อหรืออุปสรรคโดยรวม
4. ความเสี่ยง macro หลักที่ต้องระวัง
ตอบเป็นภาษาไทย กระชับ""", 700)

def news_agent(company, news):
    return run_agent(f"""คุณเป็น News Analyst วิเคราะห์ข่าว {company['ticker']}
{news}
วิเคราะห์: 1.sentiment รวม 2.ประเด็นกระทบราคา 3.ความเสี่ยงที่ต้องจับตา
ตอบเป็นภาษาไทย กระชับ""", 600)

def technical_agent(company, price_summary, indicators):
    return run_agent(f"""คุณเป็น Technical Analyst วิเคราะห์ราคา {company['ticker']}

ราคาและ Moving Averages:
{price_summary}

Technical Indicators:
{indicators}

วิเคราะห์:
1. ราคาอยู่โซนไหน (ใกล้ High/Low/กลาง)
2. แนวโน้ม MA20 vs MA50 (uptrend/downtrend/sideways)
3. RSI: overbought/oversold/neutral
4. MACD: bullish/bearish crossover
5. Stochastic: สัญญาณเข้า/ออก
6. ATR: ความผันผวนสูง/ต่ำ ควรตั้ง stop ห่างแค่ไหน
7. แนวรับ/แนวต้านหลัก
ตอบเป็นภาษาไทย กระชับ มีตัวเลขอ้างอิง""", 800)

def orchestrator_agent(company, fin, mac, geo, news, tech):
    return run_agent(f"""คุณเป็น CIO สรุปผลจากทีมผู้เชี่ยวชาญ
หุ้น: {company['ticker']} ราคา: ${company['price']:.2f} Market Cap: ${company['market_cap_b']:.1f}B

[Financial Analysis]: {fin}
[Macro Analysis]: {mac}
[Geopolitical Risk]: {geo}
[News Analysis]: {news}
[Technical Analysis]: {tech}

สรุปรวมให้ครบ:
1. ภาพรวม — น่าลงทุนมั้ย และทำไม (รวมมิติ geopolitical)
2. จุดเข้าซื้อที่แนะนำ (ราคาหรือสัญญาณ)
3. Stop Loss และเหตุผล
4. ระดับความเสี่ยงรวม (Financial + Macro + Geopolitical + Technical)
5. กลยุทธ์ที่เหมาะสม (DCA / รอ pullback / เข้าทันที / หลีกเลี่ยง)
ตอบเป็นภาษาไทย ละเอียด มีตัวเลขชัดเจน""", 3000)


def competitor_agent(company, competitors):
    """วิเคราะห์เปรียบเทียบคู่แข่ง"""
    if not competitors:
        return "ไม่พบข้อมูลคู่แข่ง"
    comp_summary = "\n".join([
        f"- {c['ticker']} ({c['name']}): "
        f"Market Cap=${c['market_cap_b']:.1f}B, "
        f"P/E={c['pe']:.1f if c['pe'] else 'N/A'}, "
        f"Gross Margin={round(c['gross_margin']*100,1) if c['gross_margin'] else 'N/A'}%, "
        f"Rev Growth={round(c['rev_growth']*100,1) if c['rev_growth'] else 'N/A'}%"
        for c in competitors
    ])
    return run_agent(f"""คุณเป็น Competitive Intelligence Analyst
เปรียบเทียบ {company['ticker']} กับคู่แข่งใน sector เดียวกัน

{company['ticker']} — Market Cap=${company['market_cap_b']:.1f}B

คู่แข่ง:
{comp_summary}

วิเคราะห์:
1. จุดแข็ง/อ่อนของ {company['ticker']} เมื่อเทียบกับคู่แข่ง
2. {company['ticker']} แพง/ถูกกว่าคู่แข่งมั้ย (relative valuation)
3. ใครเติบโตเร็วกว่าและทำไม
4. สรุป: {company['ticker']} น่าสนใจกว่าคู่แข่งมั้ย

ตอบเป็นภาษาไทย กระชับ มีตัวเลขอ้างอิง""", 1000)


def geopolitical_agent(company, geo_indicators, news):
    """วิเคราะห์ความเสี่ยงภูมิรัฐศาสตร์และสงคราม"""
    geo_str = "\n".join([f"- {k}: {v}" for k, v in geo_indicators.items()]) if geo_indicators else "ไม่มีข้อมูล"
    return run_agent(f"""คุณเป็น Geopolitical Risk Analyst
วิเคราะห์ความเสี่ยงด้านภูมิรัฐศาสตร์ สงคราม และนโยบายรัฐที่กระทบ {company['ticker']} ({company['name']})

=== ข้อมูลบริษัท ===
Sector: {company['sector']} | Industry: {company['industry']}
ธุรกิจ: {company['summary']}

=== ตัวชี้วัดความเสี่ยงตลาด (real-time) ===
{geo_str}

=== ข่าวล่าสุด ===
{news}

วิเคราะห์ให้ครบ 5 ด้าน:
1. ความเสี่ยงหลักจากสงคราม/ความขัดแย้ง (เช่น Russia-Ukraine, Israel-Gaza, US-China tension) ที่กระทบ sector นี้โดยตรง
2. VIX และ safe haven (Gold/USD) บ่งชี้ sentiment ตลาดอย่างไร
3. ความเสี่ยง Supply Chain จากความขัดแย้งโลก (วัตถุดิบ, การผลิต, logistics)
4. นโยบายรัฐบาล/การค้า (tariff, sanctions, subsidy, IRA) ที่กระทบธุรกิจนี้
5. สรุประดับความเสี่ยงภูมิรัฐศาสตร์ (สูง/กลาง/ต่ำ) พร้อมผลกระทบต่อราคาหุ้นและกลยุทธ์รับมือ

ตอบเป็นภาษาไทย ละเอียด มีเหตุผลชัดเจน""", 1000)


def chat_agent(ticker, company, fin_result, mac_result, geo_result, news_result, tech_result, final, messages, earnings_date="N/A"):
    """Chat Agent ที่รู้จักผลวิเคราะห์ทั้งหมดและราคา real-time"""
    realtime_price = get_realtime_price(ticker)

    system_context = f"""คุณเป็น Investment Advisor ผู้เชี่ยวชาญที่วิเคราะห์หุ้น {ticker} ({company['name']}) มาแล้ว
ราคาปัจจุบัน (real-time): ${realtime_price:.2f}
วันประกาศผลประกอบการถัดไป: {earnings_date}

=== ผลวิเคราะห์จากทีม ===

[Financial Analysis]
{fin_result}

[Macro Analysis]
{mac_result}

[News Analysis]
{news_result}

[Technical Analysis]
{tech_result}

[Geopolitical Risk]
{geo_result}

[CIO Summary]
{final}

=== คำแนะนำในการตอบ ===
- ตอบโดยอ้างอิงจากผลวิเคราะห์ข้างต้นเสมอ
- ใช้ราคา real-time ${realtime_price:.2f} ในการประเมินจุดเข้า/ออก
- ตอบเป็นภาษาไทย กระชับ ชัดเจน มีตัวเลขอ้างอิง
- ถ้าคำถามเกินขอบเขตข้อมูลที่มี ให้บอกตรงๆ"""

    api_messages = [{"role": "user", "content": system_context + "\n\nเริ่มการสนทนาได้เลย"}]
    api_messages.append({"role": "assistant", "content": f"พร้อมแล้วครับ ผมมีข้อมูลวิเคราะห์ {ticker} ครบถ้วน ราคาปัจจุบัน ${realtime_price:.2f} ถามได้เลยครับ"})

    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=api_messages
    )
    return response.content[0].text

# ===== UI =====
st.title("Stock Analyzer")
st.caption("Multi-Agent Analysis powered by Claude")

if "load_ticker" in st.session_state:
    ticker_input = st.session_state.pop("load_ticker")
else:
    ticker_input = st.text_input("ใส่ ticker", placeholder="เช่น BE, NVDA, TSLA").upper().strip()

if st.button("Analyze", type="primary") and ticker_input:

    if ticker_input in st.session_state["history"]:
        st.info(f"โหลดผลเก่าของ {ticker_input} (ไม่เสีย token)")
        cached      = st.session_state["history"][ticker_input]
        company     = cached["company"]
        fin_result  = cached["fin_result"]
        mac_result  = cached["mac_result"]
        geo_result  = cached.get("geo_result", "ไม่มีข้อมูล geopolitical")
        news_result = cached["news_result"]
        tech_result = cached["tech_result"]
        final       = cached["final"]
        earnings_date = "N/A"
        hist, fin, bs, cf = get_chart_data(ticker_input)

    else:
        with st.spinner("กำลังดึงข้อมูล..."):
            company = get_company_info(ticker_input)
            if not company:
                st.error(f"ไม่พบข้อมูล {ticker_input}")
                st.stop()
            financials        = get_financials(ticker_input)
            quarterly         = get_quarterly_financials(ticker_input)
            analyst           = get_analyst_ratings(ticker_input)
            earnings_date     = get_earnings_date(ticker_input)
            hist, fin, bs, cf = get_chart_data(ticker_input)
            price_summary     = get_price_summary(hist)
            indicators        = get_technical_indicators(hist)
            news              = get_news(ticker_input)
            macro             = get_macro_data()
            yield_curve       = get_yield_curve()
            geo_indicators    = get_geopolitical_indicators()

        with st.spinner("Financial Agent (+ Quarterly + Analyst)..."):
            fin_result  = financial_agent(company, financials, quarterly, analyst)
        with st.spinner("Macro Agent (+ Yield Curve)..."):
            mac_result  = macro_agent(company, macro, yield_curve)
        with st.spinner("News Agent (10 headlines)..."):
            news_result = news_agent(company, news)
        with st.spinner("Technical Agent (+ RSI/MACD/ATR/Stochastic)..."):
            tech_result = technical_agent(company, price_summary, indicators)
        with st.spinner("Geopolitical Agent (+ War/Policy/Supply Chain)..."):
            geo_result  = geopolitical_agent(company, geo_indicators, news)
        with st.spinner("Orchestrator สรุปภาพรวม (ทุก Agent)..."):
            final       = orchestrator_agent(company, fin_result, mac_result, geo_result, news_result, tech_result)

        st.session_state["history"][ticker_input] = {
            "company":    company,
            "fin_result":  fin_result,
            "mac_result":  mac_result,
            "geo_result":  geo_result,
            "news_result": news_result,
            "tech_result": tech_result,
            "final":       final,
        }

    # เริ่ม chat history ถ้ายังไม่มี
    if ticker_input not in st.session_state["chat_messages"]:
        st.session_state["chat_messages"][ticker_input] = []

    # ===== TABS =====
    tab_dash, tab_fin, tab_mac, tab_geo, tab_news, tab_tech, tab_full, tab_comp, tab_chat = st.tabs([
        "Dashboard", "Financial", "Macro", "Geopolitical", "News", "Technical", "CIO Full Report", "Competitors", "Chat"
    ])

    # ===== DASHBOARD TAB =====
    with tab_dash:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ราคา",       f"${company['price']:.2f}")
        c2.metric("Market Cap", f"${company['market_cap_b']:.1f}B")
        c3.metric("Sector",     company['sector'])
        c4.metric("Industry",   company['industry'])

        st.divider()

        st.subheader("กราฟ")
        ALL_CHARTS = [
            "ราคา + MA", "Bollinger Bands", "Trendline",
            "Volume", "RSI",
            "Revenue", "Gross Margin", "Free Cash Flow", "Debt vs Cash",
            "Macro Overlay", "Peer Comparison",
        ]
        selected_charts = st.multiselect(
            "เลือกกราฟ (เลือกได้สูงสุด 2 เพื่อดูคู่กัน)",
            options=ALL_CHARTS,
            default=["ราคา + MA", "Revenue"],
            max_selections=2,
        )
        if len(selected_charts) == 2:
            gc1, gc2 = st.columns(2)
            with gc1:
                draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf, company.get('sector','Technology'))
            with gc2:
                draw_chart(selected_charts[1], ticker_input, hist, fin, bs, cf, company.get('sector','Technology'))
        elif len(selected_charts) == 1:
            draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf, company.get('sector','Technology'))

        st.divider()

        st.subheader("สรุปภาพรวม — CIO")
        st.markdown(final[:600] + "..." if len(final) > 600 else final)
        st.info("กดแท็บ **CIO Full Report** เพื่อดูรายงานฉบับเต็ม หรือ **Chat** เพื่อถามต่อ")

        st.divider()

        st.subheader("สรุปจาก Agents")
        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Financial Agent**")
            st.markdown(fin_result[:300] + "...")
            with st.expander("ดูรายละเอียด Financial"):
                st.markdown(fin_result)

            st.markdown("**Macro Agent**")
            st.markdown(mac_result[:300] + "...")
            with st.expander("ดูรายละเอียด Macro"):
                st.markdown(mac_result)

        with a2:
            st.markdown("**Geopolitical Agent**")
            st.markdown(geo_result[:300] + "...")
            with st.expander("ดูรายละเอียด Geopolitical"):
                st.markdown(geo_result)

            st.markdown("**News Agent**")
            st.markdown(news_result[:300] + "...")
            with st.expander("ดูรายละเอียด News"):
                st.markdown(news_result)

            st.markdown("**Technical Agent**")
            st.markdown(tech_result[:300] + "...")
            with st.expander("ดูรายละเอียด Technical"):
                st.markdown(tech_result)

    # ===== DETAIL TABS =====
    with tab_fin:
        st.subheader(f"Financial Analysis — {ticker_input}")
        st.markdown(fin_result)

    with tab_mac:
        st.subheader(f"Macro Analysis — {ticker_input}")
        st.markdown(mac_result)

    with tab_geo:
        st.subheader(f"Geopolitical Risk — {ticker_input}")
        st.caption("วิเคราะห์ความเสี่ยงสงคราม นโยบายรัฐ และภูมิรัฐศาสตร์โลก")
        st.markdown(geo_result)

    with tab_news:
        st.subheader(f"News & Sentiment — {ticker_input}")
        st.markdown(news_result)

    with tab_tech:
        st.subheader(f"Technical Analysis — {ticker_input}")
        st.markdown(tech_result)

    with tab_full:
        st.subheader(f"CIO Full Report — {ticker_input}")
        st.markdown(final)
        st.divider()

        # แปลง chat history เป็น text
        chat_history_text = ""
        msgs = st.session_state["chat_messages"].get(ticker_input, [])
        if msgs:
            lines = []
            for msg in msgs:
                prefix = "User: " if msg["role"] == "user" else "Agent: "
                lines.append(f"{prefix}{msg['content']}")
            chat_history_text = "\n".join(lines)

        full_text = f"""=== {ticker_input} — {company['name']} ===

[Financial Analysis]
{fin_result}

[Macro Analysis]
{mac_result}

[Geopolitical Analysis]
{geo_result}

[News Analysis]
{news_result}

[Technical Analysis]
{tech_result}

[CIO Full Report]
{final}

[Chat History]
{chat_history_text}"""

        st.download_button(
            label="Download ผลวิเคราะห์ + บทสนทนา เป็น .txt",
            data=full_text.encode("utf-8"),
            file_name=f"{ticker_input}_analysis.txt",
            mime="text/plain",
        )
        if msgs:
            st.caption(f"ไฟล์นี้รวมบทสนทนา {len(msgs)} ข้อความไว้ด้วย Upload กลับมาคุยต่อได้เลย")

    # ===== COMPETITORS TAB =====
    with tab_comp:
        st.subheader(f"Competitor Analysis — {ticker_input}")
        st.caption(f"เปรียบเทียบกับคู่แข่งใน {company['sector']}")

        with st.spinner("กำลังดึงข้อมูลคู่แข่ง..."):
            competitors = get_competitors(ticker_input, company.get("sector", "Technology"))

        if competitors:
            # ตารางเปรียบเทียบ
            comp_data = {
                "Ticker":         [ticker_input] + [c["ticker"] for c in competitors],
                "Name":           [company["name"][:20]] + [c["name"] for c in competitors],
                "Market Cap (B$)": [company["market_cap_b"]] + [c["market_cap_b"] for c in competitors],
                "P/E":            ["N/A"] + [f"{c['pe']:.1f}" if c["pe"] else "N/A" for c in competitors],
                "Gross Margin":   ["N/A"] + [f"{round(c['gross_margin']*100,1)}%" if c["gross_margin"] else "N/A" for c in competitors],
                "Rev Growth":     ["N/A"] + [f"{round(c['rev_growth']*100,1)}%" if c["rev_growth"] else "N/A" for c in competitors],
            }
            st.dataframe(comp_data, use_container_width=True)

            # กราฟเปรียบเทียบ Market Cap
            st.divider()
            tickers_all = [ticker_input] + [c["ticker"] for c in competitors]
            caps_all    = [company["market_cap_b"]] + [c["market_cap_b"] for c in competitors]
            margins_all = [0] + [round(c["gross_margin"]*100,1) if c["gross_margin"] else 0 for c in competitors]
            colors_cap  = ["#EF9F27" if t == ticker_input else "#85B7EB" for t in tickers_all]

            gc1, gc2 = st.columns(2)
            with gc1:
                fig = go.Figure(go.Bar(x=tickers_all, y=caps_all, marker_color=colors_cap,
                                        text=[f"${v}B" for v in caps_all], textposition="outside"))
                fig.update_layout(title="Market Cap (B$)", height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with gc2:
                fig = go.Figure(go.Bar(x=tickers_all, y=margins_all,
                                        marker_color=["#EF9F27" if t == ticker_input else "#1D9E75" for t in tickers_all],
                                        text=[f"{v}%" for v in margins_all], textposition="outside"))
                fig.update_layout(title="Gross Margin (%)", height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)

            # Competitor Agent วิเคราะห์
            st.divider()
            with st.spinner("Competitor Agent กำลังวิเคราะห์..."):
                comp_analysis = competitor_agent(company, competitors)
            st.markdown(comp_analysis)
        else:
            st.info(f"ไม่พบคู่แข่งสำหรับ sector: {company.get('sector', 'N/A')}")

    # ===== CHAT TAB =====
    with tab_chat:
        st.subheader(f"Chat กับ Agent — {ticker_input}")
        st.caption(f"Agent รู้จักผลวิเคราะห์ทั้งหมดและราคา real-time ของ {ticker_input}")

        # ปุ่มล้างประวัติแชท
        if st.button("ล้างประวัติแชท", key="clear_chat"):
            st.session_state["chat_messages"][ticker_input] = []
            st.rerun()

        # แสดงประวัติแชท
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_messages"][ticker_input]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ช่องพิมพ์คำถาม
        if prompt := st.chat_input(f"ถามเกี่ยวกับ {ticker_input} ได้เลย..."):
            # เพิ่มคำถามผู้ใช้
            st.session_state["chat_messages"][ticker_input].append({
                "role": "user", "content": prompt
            })
            with st.chat_message("user"):
                st.markdown(prompt)

            # เรียก Chat Agent
            with st.chat_message("assistant"):
                with st.spinner("กำลังคิด..."):
                    reply = chat_agent(
                        ticker_input, company,
                        fin_result, mac_result, geo_result, news_result, tech_result, final,
                        st.session_state["chat_messages"][ticker_input][:-1],
                        earnings_date
                    )
                st.markdown(reply)

            # บันทึกคำตอบ
            st.session_state["chat_messages"][ticker_input].append({
                "role": "assistant", "content": reply
            })
