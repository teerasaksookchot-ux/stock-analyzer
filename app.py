import streamlit as st
import yfinance as yf
import anthropic
import os
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analyzer", layout="wide")

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

@st.cache_data(ttl=3600)
def get_chart_data(ticker):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="6mo")
    fin   = stock.financials
    bs    = stock.balance_sheet
    cf    = stock.cashflow
    return hist, fin, bs, cf

def get_price_summary(hist):
    if hist is None or hist.empty:
        return "ไม่มีข้อมูลราคา"
    latest  = hist["Close"].iloc[-1]
    high_6m = hist["Close"].max()
    low_6m  = hist["Close"].min()
    avg_vol = int(hist["Volume"].mean())
    ma20    = hist["Close"].tail(20).mean()
    ma50    = hist["Close"].tail(50).mean()
    return (f"ราคาล่าสุด: ${latest:.2f} | "
            f"6M High: ${high_6m:.2f} | 6M Low: ${low_6m:.2f} | "
            f"MA20: ${ma20:.2f} | MA50: ${ma50:.2f} | "
            f"Avg Volume: {avg_vol:,}")

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

# ========== CHART FUNCTIONS ==========
def draw_chart(name, ticker, hist, fin, bs, cf):
    if name == "ราคา + MA":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
                                  name="ราคา", line=dict(color="#378ADD", width=2)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(),
                                  name="MA20", line=dict(color="#EF9F27", dash="dash")))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),
                                  name="MA50", line=dict(color="#1D9E75", dash="dash")))
        fig.update_layout(title=f"{ticker} ราคาย้อนหลัง 6 เดือน",
                           height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Volume":
        colors = ["#E24B4A" if c < o else "#1D9E75"
                  for c, o in zip(hist["Close"], hist["Open"])]
        fig = go.Figure(go.Bar(x=hist.index, y=hist["Volume"],
                                marker_color=colors, name="Volume"))
        fig.update_layout(title="Volume", height=300,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif name == "RSI":
        delta = hist["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = 100 - (100 / (1 + rs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=rsi,
                                  name="RSI", line=dict(color="#7F77DD", width=2)))
        fig.add_hrect(y0=70, y1=100, fillcolor="#E24B4A", opacity=0.08, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="#1D9E75", opacity=0.08, line_width=0)
        fig.add_hline(y=70, line_dash="dash", line_color="#E24B4A",
                       annotation_text="Overbought 70", annotation_position="right")
        fig.add_hline(y=30, line_dash="dash", line_color="#1D9E75",
                       annotation_text="Oversold 30", annotation_position="right")
        fig.update_layout(title="RSI (14)", height=300, yaxis_range=[0, 100],
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif name == "Revenue":
        if fin is not None and "Total Revenue" in fin.index:
            rev   = fin.loc["Total Revenue"].dropna().sort_index() / 1e9
            years = [str(d.year) for d in rev.index]
            vals  = [round(v, 2) for v in rev.values]
            fig   = go.Figure(go.Bar(
                x=years, y=vals, marker_color="#378ADD",
                text=[f"${v}B" for v in vals], textposition="outside"
            ))
            fig.update_layout(title="Revenue (B$)", height=300,
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูล Revenue")

    elif name == "Gross Margin":
        if fin is not None and "Gross Profit" in fin.index and "Total Revenue" in fin.index:
            gp     = fin.loc["Gross Profit"].dropna()
            rev    = fin.loc["Total Revenue"].dropna()
            margin = (gp / rev * 100).sort_index().round(1)
            years  = [str(d.year) for d in margin.index]
            vals   = list(margin.values)
            fig    = go.Figure(go.Bar(
                x=years, y=vals, marker_color="#1D9E75",
                text=[f"{v}%" for v in vals], textposition="outside"
            ))
            fig.update_layout(title="Gross Margin (%)", height=300,
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูล Gross Margin")

    elif name == "Free Cash Flow":
        if cf is not None and "Free Cash Flow" in cf.index:
            fcf    = cf.loc["Free Cash Flow"].dropna().sort_index() / 1e6
            years  = [str(d.year) for d in fcf.index]
            vals   = [round(v, 0) for v in fcf.values]
            colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in vals]
            fig    = go.Figure(go.Bar(
                x=years, y=vals, marker_color=colors,
                text=[f"${v}M" for v in vals], textposition="outside"
            ))
            fig.add_hline(y=0, line_color="gray", line_width=0.5)
            fig.update_layout(title="Free Cash Flow (M$)", height=300,
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูล Free Cash Flow")

    elif name == "Debt vs Cash":
        if bs is not None and not bs.empty:
            years_list, cash_list, debt_list = [], [], []
            for col in sorted(bs.columns)[:4]:
                cash = 0
                debt = 0
                for key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                    if key in bs.index:
                        v = bs.loc[key, col]
                        if v and str(v) != "nan":
                            cash = float(v) / 1e9
                            break
                for key in ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]:
                    if key in bs.index:
                        v = bs.loc[key, col]
                        if v and str(v) != "nan":
                            debt = float(v) / 1e9
                            break
                years_list.append(str(col.year))
                cash_list.append(round(cash, 2))
                debt_list.append(round(debt, 2))
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Cash", x=years_list, y=cash_list,
                                  marker_color="#1D9E75",
                                  text=[f"${v}B" for v in cash_list],
                                  textposition="outside"))
            fig.add_trace(go.Bar(name="Debt", x=years_list, y=debt_list,
                                  marker_color="#E24B4A",
                                  text=[f"${v}B" for v in debt_list],
                                  textposition="outside"))
            fig.update_layout(title="Debt vs Cash (B$)", barmode="group",
                               height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูล Balance Sheet")

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

    # metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ราคา",       f"${company['price']:.2f}")
    col2.metric("Market Cap", f"${company['market_cap_b']:.1f}B")
    col3.metric("Sector",     company['sector'])
    col4.metric("Industry",   company['industry'])

    st.divider()

    # ========== CHART SELECTOR ==========
    st.subheader("กราฟ")

    ALL_CHARTS = [
        "ราคา + MA",
        "Volume",
        "RSI",
        "Revenue",
        "Gross Margin",
        "Free Cash Flow",
        "Debt vs Cash",
    ]

    selected_charts = st.multiselect(
        "เลือกกราฟที่ต้องการดู (เลือกได้สูงสุด 2 กราฟ เพื่อดูคู่กัน)",
        options=ALL_CHARTS,
        default=["ราคา + MA", "Revenue"],
        max_selections=2
    )

    if len(selected_charts) == 2:
        c1, c2 = st.columns(2)
        with c1:
            draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf)
        with c2:
            draw_chart(selected_charts[1], ticker_input, hist, fin, bs, cf)
    elif len(selected_charts) == 1:
        draw_chart(selected_charts[0], ticker_input, hist, fin, bs, cf)
    else:
        st.info("เลือกกราฟด้านบนเพื่อแสดงผล")

    st.divider()

    # ========== AGENTS ==========
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

    with st.spinner("Orchestrator กำลังสรุป..."):
        final = orchestrator_agent(company, fin_result, mac_result, news_result, tech_result)

    st.divider()
    st.subheader("สรุปรวม — CIO Report")
    st.markdown(final)
