import streamlit as st
import yfinance as yf
import anthropic
import requests
import plotly.graph_objects as go
import numpy as np
from scipy.signal import argrelextrema

st.set_page_config(page_title="Stock Analyzer", layout="wide")
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

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

def get_news(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "ไม่มีข่าว"
        return "\n".join([f"- {n.get('content', {}).get('title', 'N/A')}" for n in news[:5]])
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

def draw_chart(name, ticker, hist, fin, bs, cf):
    H = 300
    M = dict(l=0, r=0, t=40, b=0)

    if name == "ราคา + MA":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],                     name="ราคา", line=dict(color="#378ADD", width=2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(),  name="MA20", line=dict(color="#EF9F27", dash="dash")))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),  name="MA50", line=dict(color="#1D9E75", dash="dash")))
        fig.update_layout(title=f"{ticker} ราคา + MA", height=H, margin=M)
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Bollinger Bands":
        ma, upper, lower = calc_bollinger(hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=upper,  name="Upper", line=dict(color="#E24B4A", dash="dash", width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=lower,  name="Lower", line=dict(color="#1D9E75", dash="dash", width=1), fill="tonexty", fillcolor="rgba(29,158,117,0.06)"))
        fig.add_trace(go.Scatter(x=hist.index, y=ma,     name="MA20",  line=dict(color="#EF9F27", width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="ราคา", line=dict(color="#378ADD", width=2)))
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

# ===== AGENTS =====

def run_agent(prompt, max_tokens=1000):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

def financial_agent(company, financials):
    return run_agent(f"""คุณเป็น Financial Analyst วิเคราะห์งบการเงินของ {company['ticker']}
{financials['income_stmt']}
{financials['balance_sheet']}
{financials['cash_flow']}
วิเคราะห์: 1.แนวโน้มรายได้และกำไร 2.ความแข็งแกร่งงบดุล 3.คุณภาพกระแสเงินสด 4.จุดแข็ง/อ่อน
ตอบเป็นภาษาไทย กระชับ อ้างอิงตัวเลขจริง""", 1000)

def macro_agent(company, macro):
    return run_agent(f"""คุณเป็น Macro Economist วิเคราะห์ผลกระทบ macro ต่อ {company['ticker']}
Macro: {macro} | Sector: {company['sector']} | ธุรกิจ: {company['summary']}
วิเคราะห์: 1.ผลกระทบดอกเบี้ย/เงินเฟ้อ 2.macro เอื้อหรืออุปสรรค 3.ความเสี่ยง macro
ตอบเป็นภาษาไทย กระชับ""", 600)

def news_agent(company, news):
    return run_agent(f"""คุณเป็น News Analyst วิเคราะห์ข่าว {company['ticker']}
{news}
วิเคราะห์: 1.sentiment รวม 2.ประเด็นกระทบราคา 3.ความเสี่ยงที่ต้องจับตา
ตอบเป็นภาษาไทย กระชับ""", 600)

def technical_agent(company, price_summary):
    return run_agent(f"""คุณเป็น Technical Analyst วิเคราะห์ราคา {company['ticker']}
{price_summary}
วิเคราะห์: 1.ราคาอยู่โซนไหน 2.แนวโน้ม MA20 vs MA50 3.แนวรับ/แนวต้าน
ตอบเป็นภาษาไทย กระชับ""", 600)

def orchestrator_agent(company, fin, mac, news, tech):
    return run_agent(f"""คุณเป็น CIO สรุปผลจากทีม
หุ้น: {company['ticker']} ราคา: ${company['price']:.2f} Market Cap: ${company['market_cap_b']:.1f}B
[Financial]: {fin}
[Macro]: {mac}
[News]: {news}
[Technical]: {tech}
สรุป: 1.น่าลงทุนมั้ย 2.จุดเข้าซื้อ 3.Stop Loss 4.ระดับความเสี่ยง 5.กลยุทธ์
ตอบเป็นภาษาไทย ละเอียด มีตัวเลขชัดเจน""", 3000)

# ===== UI =====
st.title("Stock Analyzer")
st.caption("Multi-Agent Analysis powered by Claude")

ticker_input = st.text_input("ใส่ ticker", placeholder="เช่น BE, NVDA, TSLA").upper().strip()

if st.button("Analyze", type="primary") and ticker_input:

    with st.spinner("กำลังดึงข้อมูล..."):
        company = get_company_info(ticker_input)
        if not company:
            st.error(f"ไม่พบข้อมูล {ticker_input}")
            st.stop()
        financials        = get_financials(ticker_input)
        hist, fin, bs, cf = get_chart_data(ticker_input)
        price_summary     = get_price_summary(hist)
        news              = get_news(ticker_input)
        macro             = get_macro_data()

    with st.spinner("Financial Agent..."):
        fin_result  = financial_agent(company, financials)
    with st.spinner("Macro Agent..."):
        mac_result  = macro_agent(company, macro)
    with st.spinner("News Agent..."):
        news_result = news_agent(company, news)
    with st.spinner("Technical Agent..."):
        tech_result = technical_agent(company, price_summary)
    with st.spinner("Orchestrator สรุปภาพรวม..."):
        final       = orchestrator_agent(company, fin_result, mac_result, news_result, tech_result)

    # ===== TABS =====
    tab_dash, tab_fin, tab_mac, tab_news, tab_tech, tab_full = st.tabs([
        "Dashboard",
        "Financial",
        "Macro",
        "News",
        "Technical",
        "CIO Full Report",
    ])

    # ===== DASHBOARD TAB =====
    with tab_dash:

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ราคา",       f"${company['price']:.2f}")
        c2.metric("Market Cap", f"${company['market_cap_b']:.1f}B")
        c3.metric("Sector",     company['sector'])
        c4.metric("Industry",   company['industry'])

        st.divider()

        # Chart selector
        st.subheader("กราฟ")
        ALL_CHARTS = [
            "ราคา + MA", "Bollinger Bands", "Trendline",
            "Volume", "RSI",
            "Revenue", "Gross Margin", "Free Cash Flow", "Debt vs Cash",
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
                draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf)
            with gc2:
                draw_chart(selected_charts[1], ticker_input, hist, fin, bs, cf)
        elif len(selected_charts) == 1:
            draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf)

        st.divider()

        # CIO Summary
        st.subheader("สรุปภาพรวม — CIO")
        st.markdown(final[:600] + "..." if len(final) > 600 else final)
        st.info("กดแท็บ **CIO Full Report** ด้านบนเพื่อดูรายงานฉบับเต็ม")

        st.divider()

        # Agent quick summaries
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

    with tab_news:
        st.subheader(f"News & Sentiment — {ticker_input}")
        st.markdown(news_result)

    with tab_tech:
        st.subheader(f"Technical Analysis — {ticker_input}")
        st.markdown(tech_result)

    with tab_full:
        st.subheader(f"CIO Full Report — {ticker_input}")
        st.markdown(final)
