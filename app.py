import streamlit as st
import yfinance as yf
import anthropic
import os
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analyzer", layout="wide")

# ดึง API key จาก Streamlit Secrets
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ========== DATA NODES ==========
def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        if not info.get("currentPrice") and not info.get("regularMarketPrice"):
            return None
        return {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A")[:500],
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

def get_price_history(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty:
            return None, "ไม่มีข้อมูลราคา"
        latest  = hist["Close"].iloc[-1]
        high_6m = hist["Close"].max()
        low_6m  = hist["Close"].min()
        avg_vol = int(hist["Volume"].mean())
        ma20    = hist["Close"].tail(20).mean()
        ma50    = hist["Close"].tail(50).mean()
        summary = (f"ราคาล่าสุด: ${latest:.2f} | "
                   f"6M High: ${high_6m:.2f} | 6M Low: ${low_6m:.2f} | "
                   f"MA20: ${ma20:.2f} | MA50: ${ma50:.2f} | "
                   f"Avg Volume: {avg_vol:,}")
        return hist, summary
    except:
        return None, "ไม่มีข้อมูลราคา"

def get_news(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "ไม่มีข่าว"
        headlines = [f"- {n.get('content', {}).get('title', 'N/A')}" for n in news[:5]]
        return "\n".join(headlines)
    except:
        return "ไม่มีข่าว"

def get_macro_data():
    try:
        fed = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
            timeout=5
        ).text.strip().split("\n")[-1]
        cpi = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
            timeout=5
        ).text.strip().split("\n")[-1]
        fed_date, fed_rate = fed.split(",")
        cpi_date, cpi_val  = cpi.split(",")
        return f"Fed Rate: {fed_rate}% (ณ {fed_date}) | CPI: {cpi_val} (ณ {cpi_date})"
    except:
        return "ไม่สามารถดึงข้อมูล macro ได้"

# ========== AGENTS ==========
def run_agent(prompt, max_tokens=1000):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

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

# ========== UI ==========
st.title("Stock Analyzer")
st.caption("Multi-Agent Analysis powered by Claude")

ticker_input = st.text_input("ใส่ ticker", placeholder="เช่น BE, NVDA, TSLA").upper().strip()

if st.button("Analyze", type="primary") and ticker_input:
    # ดึงข้อมูล
    with st.spinner("กำลังดึงข้อมูล..."):
        company  = get_company_info(ticker_input)
        if not company:
            st.error(f"ไม่พบข้อมูล {ticker_input}")
            st.stop()

        financials    = get_financials(ticker_input)
        hist, price_summary = get_price_history(ticker_input)
        news          = get_news(ticker_input)
        macro         = get_macro_data()

    # metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ราคา", f"${company['price']:.2f}")
    col2.metric("Market Cap", f"${company['market_cap_b']:.1f}B")
    col3.metric("Sector", company['sector'])
    col4.metric("Industry", company['industry'])

    # กราฟราคา
    if hist is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
                                  name="ราคา", line=dict(color="#378ADD", width=2)))
        ma20 = hist["Close"].rolling(20).mean()
        ma50 = hist["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=ma20,
                                  name="MA20", line=dict(color="#EF9F27", dash="dash")))
        fig.add_trace(go.Scatter(x=hist.index, y=ma50,
                                  name="MA50", line=dict(color="#1D9E75", dash="dash")))
        fig.update_layout(title=f"{ticker_input} ราคาย้อนหลัง 6 เดือน",
                           height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # รัน agents
    st.subheader("ผลวิเคราะห์จาก Agents")
    col_a, col_b = st.columns(2)

    with col_a:
        with st.spinner("Financial Agent..."):
            fin_result = financial_agent(company, financials)
        with st.expander("Financial Agent", expanded=True):
            st.markdown(fin_result)

        with st.spinner("Macro Agent..."):
            mac_result = macro_agent(company, macro)
        with st.expander("Macro Agent", expanded=True):
            st.markdown(mac_result)

    with col_b:
        with st.spinner("News Agent..."):
            news_result = news_agent(company, news)
        with st.expander("News Agent", expanded=True):
            st.markdown(news_result)

        with st.spinner("Technical Agent..."):
            tech_result = technical_agent(company, price_summary)
        with st.expander("Technical Agent", expanded=True):
            st.markdown(tech_result)

    # orchestrator
    with st.spinner("Orchestrator กำลังสรุป..."):
        final = orchestrator_agent(company, fin_result, mac_result, news_result, tech_result)

    st.divider()
    st.subheader("สรุปรวม — CIO Report")
    st.markdown(final)
