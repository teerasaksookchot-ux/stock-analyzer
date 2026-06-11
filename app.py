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
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Stock Analyzer", layout="wide")
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


# ===== SUPABASE DATABASE =====
@st.cache_resource
def get_supabase():
    """เชื่อมต่อ Supabase — คืน None ถ้าไม่ได้ตั้งค่า"""
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except:
        return None

def db_save_analysis(ticker, company, fin, mac, geo, insider, news, tech, final):
    db = get_supabase()
    if not db:
        return
    try:
        db.table("analyses").delete().eq("ticker", ticker).execute()
        db.table("analyses").insert({
            "ticker":         ticker,
            "company_name":   company.get("name", ""),
            "price":          company.get("price", 0),
            "market_cap_b":   company.get("market_cap_b", 0),
            "sector":         company.get("sector", ""),
            "industry":       company.get("industry", ""),
            "fin_result":     fin,
            "mac_result":     mac,
            "geo_result":     geo,
            "insider_result": insider,
            "news_result":    news,
            "tech_result":    tech,
            "final":          final,
            "analyzed_at":    datetime.now().isoformat(),
        }).execute()
    except Exception as e:
        st.warning(f"DB save error: {e}")

def db_load_analysis(ticker) -> dict | None:
    db = get_supabase()
    if not db:
        return None
    try:
        r = db.table("analyses").select("*").eq("ticker", ticker).limit(1).execute()
        return r.data[0] if r.data else None
    except:
        return None

def db_load_all_tickers() -> list:
    db = get_supabase()
    if not db:
        return []
    try:
        r = db.table("analyses").select("ticker, company_name, price, analyzed_at").order("analyzed_at", desc=True).execute()
        return r.data or []
    except:
        return []

def db_save_chat(ticker, messages):
    db = get_supabase()
    if not db:
        return
    try:
        db.table("chat_messages").delete().eq("ticker", ticker).execute()
        if messages:
            db.table("chat_messages").insert([
                {"ticker": ticker, "role": m["role"], "content": m["content"]}
                for m in messages
            ]).execute()
    except:
        pass

def db_load_chat(ticker) -> list:
    db = get_supabase()
    if not db:
        return []
    try:
        r = db.table("chat_messages").select("role, content").eq("ticker", ticker).order("id").execute()
        return [{"role": x["role"], "content": x["content"]} for x in r.data] if r.data else []
    except:
        return []

def db_save_alerts(alerts):
    db = get_supabase()
    if not db:
        return
    try:
        db.table("alerts").delete().eq("is_active", True).execute()
        if alerts:
            db.table("alerts").insert([{
                "ticker":       a["ticker"],
                "target_price": a["target"],
                "direction":    a["direction"],
                "label":        a.get("label", ""),
                "is_active":    True,
            } for a in alerts]).execute()
    except:
        pass

def db_load_alerts() -> list:
    db = get_supabase()
    if not db:
        return []
    try:
        r = db.table("alerts").select("*").eq("is_active", True).execute()
        return [{"ticker": x["ticker"], "target": x["target_price"],
                 "direction": x["direction"], "label": x["label"]}
                for x in r.data] if r.data else []
    except:
        return []

def db_connected() -> bool:
    return get_supabase() is not None

# ===== SESSION CACHE =====
if "history" not in st.session_state:
    st.session_state["history"] = {}
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = {}
if "alerts" not in st.session_state:
    st.session_state["alerts"] = db_load_alerts()
if "db_history_loaded" not in st.session_state:
    st.session_state["db_history_loaded"] = False
if "show_portfolio" not in st.session_state:
    st.session_state["show_portfolio"] = False
if "news_filter" not in st.session_state:
    st.session_state["news_filter"] = "All"
if "news_search_query" not in st.session_state:
    st.session_state["news_search_query"] = ""

# ===== PARSE UPLOADED FILE =====
def parse_analysis_file(text: str) -> dict | None:
    try:
        markers = {
            "[Financial Analysis]":    "fin_result",
            "[Macro Analysis]":        "mac_result",
            "[Geopolitical Analysis]": "geo_result",
            "[Insider Analysis]":      "insider_result",
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
            "company":        company,
            "fin_result":     sections["fin_result"],
            "mac_result":     sections["mac_result"],
            "geo_result":     sections["geo_result"],
            "insider_result": sections["insider_result"],
            "news_result":    sections["news_result"],
            "tech_result":    sections["tech_result"],
            "final":          sections["final"],
            "chat_messages":  chat_messages,
        }
    except:
        return None

# ===== SIDEBAR =====
with st.sidebar:

    # ===== DB STATUS =====
    if db_connected():
        st.success("🗄 Database เชื่อมต่อแล้ว", icon="✅")
    else:
        st.warning("🗄 Database ยังไม่ได้ตั้งค่า", icon="⚠️")

    st.divider()

    # ===== DB HISTORY =====
    st.subheader("📋 ประวัติทั้งหมด (Database)")
    if db_connected():
        db_tickers = db_load_all_tickers()
        if db_tickers:
            for row in db_tickers:
                t   = row["ticker"]
                dt  = row.get("analyzed_at", "")[:10]
                px  = row.get("price", 0)
                col1, col2 = st.sidebar.columns([3, 1])
                if col1.button(f"{t}  ${px:.0f}  {dt}", key=f"dbload_{t}"):
                    st.session_state["load_ticker"] = t
                if col2.button("✕", key=f"dbdel_{t}"):
                    get_supabase().table("analyses").delete().eq("ticker", t).execute()
                    get_supabase().table("chat_messages").delete().eq("ticker", t).execute()
                    st.rerun()
        else:
            st.caption("ยังไม่มีประวัติใน Database")
    else:
        st.caption("ตั้งค่า Supabase เพื่อดูประวัติถาวร")
        if st.session_state["history"]:
            st.caption("─── Session (หายเมื่อปิด browser) ───")
            for saved_ticker in list(st.session_state["history"].keys()):
                col1, col2 = st.sidebar.columns([3, 1])
                if col1.button(saved_ticker, key=f"load_{saved_ticker}"):
                    st.session_state["load_ticker"] = saved_ticker
                if col2.button("✕", key=f"del_{saved_ticker}"):
                    del st.session_state["history"][saved_ticker]
                    st.session_state["chat_messages"].pop(saved_ticker, None)
                    st.rerun()

    st.divider()

    # ===== FILE UPLOAD (แยกส่วนชัดเจน) =====
    st.subheader("📂 อัปโหลดไฟล์เก่า")
    st.caption("สำหรับไฟล์ .txt ที่เคย Download ไว้")
    uploaded = st.file_uploader(
        "เลือกไฟล์ .txt",
        type=["txt"],
        help="ไฟล์จากปุ่ม Download ใน CIO Full Report",
        label_visibility="collapsed",
    )
    if uploaded:
        text   = uploaded.read().decode("utf-8")
        parsed = parse_analysis_file(text)
        if parsed:
            t = parsed["company"]["ticker"]
            st.session_state["history"][t] = parsed
            if parsed.get("chat_messages"):
                st.session_state["chat_messages"][t] = parsed["chat_messages"]
                st.success(f"โหลด {t} สำเร็จ ({len(parsed['chat_messages'])} ข้อความ)")
            else:
                st.success(f"โหลด {t} สำเร็จ")
            st.session_state["load_ticker"] = t
            st.rerun()
        else:
            st.error("ไฟล์ไม่ถูกต้อง — ใช้ได้เฉพาะไฟล์จากปุ่ม Download")

    st.divider()

    # ===== ALERT SECTION =====
    st.subheader("⚡ Price Alert")

    has_telegram = bool(st.secrets.get("TELEGRAM_TOKEN")) and bool(st.secrets.get("TELEGRAM_CHAT_ID"))
    has_tavily = bool(st.secrets.get("TAVILY_API_KEY"))
    if has_tavily:
        st.caption("Tavily ✅ ข่าวสดเปิดใช้งานแล้ว")
    else:
        st.caption("Tavily ⚠️ ยังไม่ได้ตั้งค่า (ใช้ Yahoo News แทน) — สมัครฟรีที่ app.tavily.com แล้วใส่ TAVILY_API_KEY ใน Secrets")

    if not has_telegram:
        with st.expander("ตั้งค่า Telegram ก่อนใช้งาน"):
            st.caption("""1. สร้าง bot ที่ @BotFather → /newbot\n2. Copy token ที่ได้\n3. ไปที่ Streamlit Cloud → Settings → Secrets ใส่:\n   TELEGRAM_TOKEN = "your_token"\n   TELEGRAM_CHAT_ID = "your_chat_id"\n4. หา chat_id: เปิด t.me/userinfobot แล้วกด Start""")

    with st.form("alert_form", clear_on_submit=True):
        a_ticker    = st.text_input("Ticker", placeholder="เช่น IONQ").upper().strip()
        a_target    = st.number_input("ราคาเป้าหมาย $", min_value=0.01, step=0.5)
        a_direction = st.selectbox("เมื่อราคา", ["ลงถึง (Buy Zone) ↓", "ขึ้นถึง (Take Profit) ↑"])
        a_label     = st.text_input("หมายเหตุ (เช่น Zone A)", placeholder="ไม่บังคับ")
        submitted   = st.form_submit_button("➕ ตั้ง Alert", use_container_width=True)
        if submitted and a_ticker and a_target > 0:
            direction = "below" if "ลงถึง" in a_direction else "above"
            new_alert = {
                "ticker":    a_ticker,
                "target":    a_target,
                "direction": direction,
                "label":     a_label or f"{a_ticker} @ ${a_target}",
            }
            st.session_state["alerts"].append(new_alert)
            db_save_alerts(st.session_state["alerts"])
            st.success(f"ตั้ง Alert {a_ticker} @ ${a_target} แล้ว")

    # แสดง alerts ที่ตั้งไว้
    if st.session_state["alerts"]:
        st.caption(f"Alerts ที่ตั้งไว้ {len(st.session_state['alerts'])} รายการ")
        for i, al in enumerate(st.session_state["alerts"]):
            arrow = "↓" if al["direction"] == "below" else "↑"
            c1, c2 = st.columns([4, 1])
            c1.caption(f"{al['ticker']} {arrow} ${al['target']} — {al['label']}")
            if c2.button("✕", key=f"del_alert_{i}"):
                st.session_state["alerts"].pop(i)
                db_save_alerts(st.session_state["alerts"])
                st.rerun()

        if st.button("🔔 เช็ค Alert ทั้งหมด", use_container_width=True):
            triggered = check_alerts()
            if triggered:
                for t in triggered:
                    arrow  = "ลงถึง" if t["direction"] == "below" else "ขึ้นถึง"
                    msg    = (f"🔔 <b>Stock Alert!</b>\n"
                              f"<b>{t['ticker']}</b> {arrow} เป้า ${t['target']:.2f}\n"
                              f"ราคาปัจจุบัน: <b>${t['current_price']:.2f}</b>\n"
                              f"หมายเหตุ: {t['label']}")
                    ok = send_telegram(msg)
                    if ok:
                        st.success(f"ส่ง Telegram แล้ว: {t['ticker']} @ ${t['current_price']:.2f}")
                    else:
                        st.warning(f"{t['ticker']} ถึงเป้า ${t['target']:.2f} แต่ส่ง Telegram ไม่ได้ (ตรวจ secrets)")
            else:
                st.info("ยังไม่มี alert ที่ถึงเป้า")

    st.divider()
    st.caption("ปิด browser = ประวัติหาย\nกด Download เพื่อเก็บไฟล์ไว้")


# ===== PORTFOLIO DATABASE (ใช้ requests โดยตรง) =====
def _sb_headers() -> dict:
    key = st.secrets.get("SUPABASE_KEY", "")
    return {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

def _sb_url(table: str) -> str:
    base = st.secrets.get("SUPABASE_URL", "").rstrip("/")
    return f"{base}/rest/v1/{table}"

def portfolio_add_position(ticker, company_name, shares, entry_price, entry_date, notes=""):
    try:
        r = requests.post(
            _sb_url("portfolios"),
            headers=_sb_headers(),
            json={
                "ticker":       ticker.upper(),
                "company_name": str(company_name),
                "shares":       float(shares),
                "entry_price":  float(entry_price),
                "entry_date":   str(entry_date),
                "notes":        str(notes),
            },
            timeout=10,
        )
        if r.status_code in (200, 201):
            return True
        st.error(f"เพิ่ม position ไม่ได้: {r.json()}")
        return False
    except Exception as e:
        st.error(f"เพิ่ม position ไม่ได้: {e}")
        return False

def portfolio_delete_position(position_id):
    try:
        requests.delete(
            _sb_url("portfolios"),
            headers=_sb_headers(),
            params={"id": f"eq.{position_id}"},
            timeout=10,
        )
    except:
        pass

def portfolio_load_positions() -> list:
    try:
        r = requests.get(
            _sb_url("portfolios"),
            headers=_sb_headers(),
            params={"order": "created_at.asc"},
            timeout=10,
        )
        return r.json() if r.status_code == 200 else []
    except:
        return []

def portfolio_add_transaction(ticker, action, shares, price):
    try:
        requests.post(
            _sb_url("portfolio_transactions"),
            headers=_sb_headers(),
            json={
                "ticker": ticker.upper(),
                "action": action,
                "shares": float(shares),
                "price":  float(price),
                "amount": round(float(shares) * float(price), 2),
            },
            timeout=10,
        )
    except:
        pass

def portfolio_load_transactions() -> list:
    try:
        r = requests.get(
            _sb_url("portfolio_transactions"),
            headers=_sb_headers(),
            params={"order": "created_at.desc", "limit": "50"},
            timeout=10,
        )
        return r.json() if r.status_code == 200 else []
    except:
        return []

@st.cache_data(ttl=3600)
def get_usd_thb_rate() -> float:
    """ดึง USD/THB exchange rate"""
    try:
        rate = yf.Ticker("USDTHB=X").fast_info.last_price
        return round(rate, 2) if rate and rate > 0 else 32.84
    except:
        return 32.84

BADGE_COLORS = [
    ("#E6F1FB","#185FA5"), ("#E1F5EE","#0F6E56"), ("#EEEDFE","#3C3489"),
    ("#FAEEDA","#854F0B"), ("#FCEBEB","#A32D2D"), ("#EAF3DE","#3B6D11"),
    ("#FBEAF0","#993556"), ("#E1F5EE","#085041"), ("#FAEEDA","#633806"),
]

@st.cache_data(ttl=300)
def portfolio_get_current_prices(tickers: tuple) -> dict:
    """ดึงราคาปัจจุบันของหุ้นทั้งหมดใน portfolio — ใช้ history เพื่อความ reliable"""
    prices = {}
    for t in tickers:
        try:
            stock  = yf.Ticker(t)
            price  = prev = 0

            # 1) ลอง fast_info ก่อน (เร็วที่สุด)
            try:
                fi    = stock.fast_info
                price = float(fi.last_price or 0)
                prev  = float(fi.previous_close or price)
            except:
                pass

            # 2) fallback → history 5 วัน (reliable ที่สุด)
            if not price:
                hist   = stock.history(period="5d")
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    price = float(closes.iloc[-1])
                    prev  = float(closes.iloc[-2])
                elif len(closes) == 1:
                    price = float(closes.iloc[-1])
                    prev  = price

            # 3) fallback → info (ช้าที่สุด)
            if not price:
                info  = stock.info
                price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
                prev  = float(info.get("previousClose") or price)

            prices[t] = {
                "price":   round(price, 2),
                "prev":    round(prev, 2),
                "day_chg": round((price - prev) / prev * 100, 2) if prev else 0,
                "name":    t,
            }
        except:
            prices[t] = {"price": 0, "prev": 0, "day_chg": 0, "name": t}
    return prices

def portfolio_calc_summary(positions: list, prices: dict) -> dict:
    """คำนวณสรุป portfolio"""
    total_cost  = 0
    total_value = 0
    total_day_chg = 0

    for p in positions:
        t     = p["ticker"]
        cost  = p["shares"] * p["entry_price"]
        value = p["shares"] * prices.get(t, {}).get("price", p["entry_price"])
        prev_value = p["shares"] * prices.get(t, {}).get("prev", p["entry_price"])
        total_cost    += cost
        total_value   += value
        total_day_chg += (value - prev_value)

    unrealized_pl  = total_value - total_cost
    unrealized_pct = round(unrealized_pl / total_cost * 100, 2) if total_cost else 0
    day_chg_pct    = round(total_day_chg / (total_value - total_day_chg) * 100, 2) if total_value else 0

    return {
        "total_value":    round(total_value, 2),
        "total_cost":     round(total_cost, 2),
        "unrealized_pl":  round(unrealized_pl, 2),
        "unrealized_pct": unrealized_pct,
        "day_change":     round(total_day_chg, 2),
        "day_change_pct": day_chg_pct,
    }


# ===== PORTFOLIO NEWS =====
BULLISH_WORDS = ["beat","beats","record","contract","partnership","approved",
                  "growth","profit","surge","rally","upgrade","buy","strong",
                  "win","award","launch","breakthrough","revenue","earnings beat",
                  "raised","raised guidance","outperform","bullish","positive"]
BEARISH_WORDS = ["miss","probe","fine","investigation","lawsuit","loss",
                  "downgrade","sell","weak","decline","cut","layoff","fail",
                  "warning","recall","fraud","violation","penalty","bearish",
                  "negative","debt","bankruptcy","losses"]

import re as _re
from datetime import datetime as _dt, timedelta as _td

# titles ที่เป็น page navigation ไม่ใช่ข่าว
_NON_NEWS_TITLES = [
    "stock info", "investor relations", "news -", "- news",
    "stock price", "stock quote", "historical data", "price forecast",
    "read more", "stock analysis -", "quote &", "price &",
]

def _is_news_article(title: str, content: str) -> bool:
    """กรอง page ที่ไม่ใช่ข่าวจริงๆ ออก"""
    title_lower = title.lower()
    if any(bad in title_lower for bad in _NON_NEWS_TITLES):
        return False
    # content ที่เป็น table (มี | เยอะ) หรือ navigation
    pipe_count = content.count("|")
    if pipe_count > 5:
        return False
    return True

def _clean_content(text: str) -> str:
    """ตัด noise ทุกประเภทออก"""
    # ลบ image markdown
    text = _re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # แปลง [text](url) → text
    text = _re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # ลบ URL
    text = _re.sub(r'https?://\S+', '', text)
    # ลบ (BUSINESS WIRE)--, (PR NEWSWIRE)-- prefix
    text = _re.sub(r'\([A-Z ]+\)--', '', text)
    # ลบ NYSE: IONQ type
    text = _re.sub(r'[A-Z]+:[A-Z]+,?\s?', '', text)
    # ลบ table rows
    text = _re.sub(r'\|[^\n]+\|', '', text)
    # ลบ "Read More" navigation
    text = _re.sub(r'Read More\s*', '', text, flags=_re.IGNORECASE)
    # ลบบรรทัดที่ขึ้นต้นด้วย # * - (navigation/sidebar)
    lines = text.split('.')
    clean_lines = []
    skip_words = ["best gift", "home page", "search for symbol",
                  "power to investor", "sign in", "subscribe",
                  "advertisement", "cookie", "privacy policy"]
    for line in lines:
        stripped = line.strip()
        # ข้าม bullet points และ sidebar content
        if stripped.startswith(("#","*","- Best","- How","- What")):
            continue
        if any(s in stripped.lower() for s in skip_words):
            continue
        if len(stripped) > 15:   # ข้าม fragment สั้นเกิน
            clean_lines.append(stripped)
    text = ". ".join(clean_lines[:3])  # เอาแค่ 3 ประโยคแรก
    text = _re.sub(r'\s+', ' ', text).strip()
    return text[:220]

def _is_recent(pub_date: str, max_days: int = 30) -> bool:
    """เช็กว่าข่าวอายุไม่เกิน max_days"""
    if not pub_date:
        return True  # ไม่มีวันที่ → ผ่าน
    try:
        d = _dt.strptime(pub_date[:10], "%Y-%m-%d")
        return (_dt.now() - d).days <= max_days
    except:
        return True

def classify_sentiment(text: str) -> tuple[str, str]:
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    if bull > bear:
        return "Bullish 📈", "var(--color-text-success)"
    elif bear > bull:
        return "Bearish 📉", "var(--color-text-danger)"
    return "Neutral", "var(--color-text-secondary)"

def _format_article(a: dict, ticker: str = "") -> dict | None:
    """แปลง raw article → cleaned dict, คืน None ถ้าข่าวเก่าหรือไม่ใช่ข่าวจริง"""
    pub     = a.get("published_date", "")[:10]
    if not _is_recent(pub, max_days=30):
        return None
    title   = _clean_content(a.get("title", "N/A"))
    content = _clean_content(a.get("content", ""))
    if not title or title == "N/A" or len(title) < 10:
        return None
    if not _is_news_article(title, content):
        return None
    sentiment, color = classify_sentiment(title + " " + content)
    return {
        "ticker":     ticker,
        "title":      title,
        "content":    content,
        "url":        a.get("url", ""),
        "published":  pub,
        "sentiment":  sentiment,
        "sent_color": color,
    }

@st.cache_data(ttl=900)
def get_portfolio_news(tickers: tuple, max_per: int = 7) -> list:
    results    = []
    tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    for t in tickers:
        try:
            if tavily_key:
                from tavily import TavilyClient
                client   = TavilyClient(api_key=tavily_key)
                response = client.search(
                    query=f"{t} stock news",
                    search_depth="basic",
                    topic="news",
                    max_results=max_per,
                    days=30,
                )
                for a in response.get("results", []):
                    item = _format_article(a, t)
                    if item:
                        results.append(item)
            else:
                news = yf.Ticker(t).news or []
                for n in news[:max_per]:
                    title = _clean_content(n.get("content", {}).get("title", "N/A"))
                    sentiment, color = classify_sentiment(title)
                    results.append({
                        "ticker": t, "title": title, "content": "",
                        "url": n.get("content",{}).get("canonicalUrl",{}).get("url",""),
                        "published": "", "sentiment": sentiment, "sent_color": color,
                    })
        except:
            pass
    return results

@st.cache_data(ttl=600)
def search_news_manual(query: str, max_results: int = 10) -> list:
    tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    if not tavily_key:
        return []
    try:
        from tavily import TavilyClient
        client   = TavilyClient(api_key=tavily_key)
        response = client.search(
                query=query,
                search_depth="basic",
                topic="news",
                max_results=max_results,
                days=60,
            )
        results  = []
        for a in response.get("results", []):
            item = _format_article(a)
            if item:
                results.append(item)
        return results
    except:
        return []


# ===== AGENT CHECKPOINTS =====
def save_checkpoint(ticker: str, agent: str, result: str):
    try:
        r = requests.post(
            _sb_url("agent_checkpoints"),
            headers=_sb_headers(),
            json={"ticker": ticker, "agent": agent, "result": result},
            timeout=8,
        )
    except:
        pass

def load_checkpoints(ticker: str) -> dict:
    try:
        r = requests.get(
            _sb_url("agent_checkpoints"),
            headers=_sb_headers(),
            params={"ticker": f"eq.{ticker}", "order": "created_at.desc"},
            timeout=8,
        )
        if r.status_code == 200:
            rows = r.json()
            latest = {}
            for row in rows:
                if row["agent"] not in latest:
                    latest[row["agent"]] = row["result"]
            return latest
        return {}
    except:
        return {}

def clear_checkpoints(ticker: str):
    try:
        requests.delete(
            _sb_url("agent_checkpoints"),
            headers=_sb_headers(),
            params={"ticker": f"eq.{ticker}"},
            timeout=8,
        )
    except:
        pass


# ===== ALERT SYSTEM =====

def send_telegram(message: str) -> bool:
    """ส่ง alert ผ่าน Telegram Bot"""
    try:
        token   = st.secrets.get("TELEGRAM_TOKEN", "")
        chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return False
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=5)
        return resp.status_code == 200
    except:
        return False

def check_alerts() -> list[dict]:
    """เช็ค alerts ทั้งหมดและคืน list ที่ถึงเป้า"""
    triggered = []
    alerts    = st.session_state.get("alerts", [])
    for alert in alerts:
        try:
            price = get_realtime_price(alert["ticker"])
            if price <= 0:
                continue
            hit = (alert["direction"] == "below" and price <= alert["target"]) or                   (alert["direction"] == "above" and price >= alert["target"])
            if hit:
                triggered.append({**alert, "current_price": price})
        except:
            pass
    return triggered


# ===== INSIDER & SHORT INTEREST =====

@st.cache_data(ttl=3600)
def get_insider_short_data(ticker):
    """ดึงข้อมูล insider trading และ short interest"""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Short Interest
        short_pct   = info.get("shortPercentOfFloat", 0) or 0
        short_ratio = info.get("shortRatio", 0) or 0
        shares_short = info.get("sharesShort", 0) or 0

        # Insider Transactions
        try:
            insider_tx = stock.insider_transactions
            if insider_tx is not None and not insider_tx.empty:
                insider_str = insider_tx.head(10).to_string()
            else:
                insider_str = "ไม่มีข้อมูล"
        except:
            insider_str = "ไม่มีข้อมูล"

        # Insider Purchases Summary
        try:
            insider_buy = stock.insider_purchases
            insider_buy_str = insider_buy.to_string() if insider_buy is not None and not insider_buy.empty else "ไม่มีข้อมูล"
        except:
            insider_buy_str = "ไม่มีข้อมูล"

        # Institutional Holders
        try:
            inst = stock.institutional_holders
            inst_str = inst.head(10).to_string() if inst is not None and not inst.empty else "ไม่มีข้อมูล"
        except:
            inst_str = "ไม่มีข้อมูล"

        return {
            "short_pct_float": round(short_pct * 100, 1),
            "short_ratio":     round(short_ratio, 1),
            "shares_short":    shares_short,
            "insider_tx":      insider_str,
            "insider_buy":     insider_buy_str,
            "institutional":   inst_str,
        }
    except:
        return None

def insider_agent(company, insider_data):
    """วิเคราะห์ insider trading และ short interest"""
    if not insider_data:
        return "ไม่มีข้อมูล insider"
    return run_agent(f"""คุณเป็น Insider & Market Structure Analyst
วิเคราะห์ข้อมูล insider trading และ short interest ของ {company['ticker']} ({company['name']})

=== Short Interest ===
Short % of Float: {insider_data['short_pct_float']}%
Short Ratio (days to cover): {insider_data['short_ratio']} วัน
Shares Short: {insider_data['shares_short']:,}

=== Insider Transactions (ล่าสุด 10 รายการ) ===
{insider_data['insider_tx']}

=== Insider Buy/Sell Summary ===
{insider_data['insider_buy']}

=== Institutional Holders (Top 10) ===
{insider_data['institutional']}

วิเคราะห์:
1. Insider ซื้อหรือขายสุทธิ — สัญญาณบวก/ลบ
2. Short % of Float สูงมั้ย — ความเสี่ยง Short Squeeze
3. สถาบันใหญ่ถือมากน้อยแค่ไหน — confidence ของ smart money
4. สรุป: ภาพรวม insider และ market structure บ่งชี้อะไร

ตอบเป็นภาษาไทย กระชับ มีตัวเลขอ้างอิง""", 800)

# ===== DATA NODES =====

@st.cache_data(ttl=1800)
def get_company_info(ticker):
    """ดึง company info ด้วย Ticker object เดียว — ลด API calls"""
    try:
        stock = yf.Ticker(ticker)
        price = 0
        info  = {}

        # ดึง info ก่อน (ครั้งเดียว ใช้ Ticker object เดิม)
        try:
            info  = stock.info
            price = float(info.get("currentPrice") or
                          info.get("regularMarketPrice") or 0)
        except:
            pass

        # Fallback 1: fast_info
        if not price:
            try:
                price = float(stock.fast_info.last_price or 0)
            except:
                pass

        # Fallback 2: history (ไม่ต้องสร้าง Ticker ใหม่)
        if not price:
            try:
                h = stock.history(period="5d")
                closes = h["Close"].dropna()
                if not closes.empty:
                    price = float(closes.iloc[-1])
            except:
                pass

        # Fallback 3: yf.download — endpoint ต่างกัน
        if not price:
            try:
                d = yf.download(ticker, period="5d",
                                progress=False, auto_adjust=True,
                                actions=False)
                if not d.empty:
                    closes = d["Close"].dropna()
                    if not closes.empty:
                        price = float(closes.iloc[-1])
            except:
                pass

        if not price:
            return None

        name = (info.get("shortName") or info.get("longName") or ticker)

        # Market cap fallback จาก fast_info
        mktcap = info.get("marketCap", 0) or 0
        if not mktcap:
            try:
                mktcap = float(stock.fast_info.market_cap or 0)
            except:
                mktcap = 0

        return {
            "ticker":       ticker,
            "name":         name,
            "price":        round(price, 2),
            "market_cap_b": round(mktcap / 1e9, 1),
            "sector":       info.get("sector") or "N/A",
            "industry":     info.get("industry") or "N/A",
            "summary":      (info.get("longBusinessSummary") or "N/A")[:500],
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
    closes = hist["Close"].dropna()
    if closes.empty:
        return "ไม่มีข้อมูลราคา"
    latest = closes.iloc[-1]
    return (f"ราคาล่าสุด: ${latest:.2f} | "
            f"6M High: ${closes.max():.2f} | "
            f"6M Low: ${closes.min():.2f} | "
            f"MA20: ${closes.tail(20).mean():.2f} | "
            f"MA50: ${closes.tail(50).mean():.2f} | "
            f"Avg Volume: {int(hist['Volume'].mean()):,}")

def get_realtime_price(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("currentPrice") or info.get("regularMarketPrice", 0)
    except:
        return 0

def get_news_yfinance(ticker):
    """fallback: ดึงข่าวจาก Yahoo Finance"""
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return "ไม่มีข่าว (Yahoo Finance)"
        return "\n".join([f"- {n.get('content', {}).get('title', 'N/A')}" for n in news[:5]])
    except:
        return "ไม่มีข่าว"

def get_news(ticker, company_name=""):
    """ดึงข่าวจาก Tavily (ถ้ามี key) ไม่งั้น fallback Yahoo Finance"""
    tavily_key = st.secrets.get("TAVILY_API_KEY", "")
    if not tavily_key:
        return get_news_yfinance(ticker)
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=tavily_key)
        results = client.search(
            query=f"{ticker} {company_name} stock news latest 2025",
            search_depth="basic",
            max_results=10,
        )
        articles = results.get("results", [])
        if not articles:
            return get_news_yfinance(ticker)
        news_text = "\n".join([
            f"- [{a.get('title','N/A')}] {a.get('content','')[:300]}"
            for a in articles
        ])
        return f"[Tavily — {len(articles)} บทความ]\n{news_text}"
    except:
        return get_news_yfinance(ticker)

def get_macro_data():
    """ดึง macro indicators ครบชุดจาก FRED"""
    series = {
        "FEDFUNDS": "Fed Rate (%)",
        "CPIAUCSL": "CPI",
        "GDP":      "GDP (B$)",
        "UNRATE":   "Unemployment (%)",
        "UMCSENT":  "Consumer Sentiment",
        "NAPM":     "Manufacturing PMI",
    }
    results = []
    for sid, name in series.items():
        try:
            rows = requests.get(
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                timeout=5
            ).text.strip().split("\n")
            last = rows[-1].split(",")
            if len(last) == 2 and last[1] not in (".", ""):
                results.append(f"{name}: {last[1]} (ณ {last[0][:7]})")
        except:
            pass
    return " | ".join(results) if results else "ไม่สามารถดึงข้อมูล macro ได้"


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
            hist   = yf.Ticker(sym).history(period="5d")
            closes = hist["Close"].dropna() if not hist.empty else pd.Series()
            if len(closes) < 2:
                continue
            cur  = closes.iloc[-1]
            prev = closes.iloc[0]
            if prev == 0 or str(cur) == "nan" or str(prev) == "nan":
                continue
            chg  = round((cur - prev) / prev * 100, 1)
            sign = "+" if chg >= 0 else ""
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

# ===== SEC EDGAR (Official Financial Data) =====
@st.cache_data(ttl=86400)
def get_sec_financials(ticker: str) -> str:
    """ดึงงบการเงิน official จาก SEC EDGAR"""
    try:
        headers = {"User-Agent": "StockAnalyzer app@stockanalyzer.com"}

        # หา CIK จาก ticker
        tickers_data = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers, timeout=10
        ).json()

        cik = None
        for _, val in tickers_data.items():
            if val.get("ticker", "").upper() == ticker.upper():
                cik = str(val["cik_str"]).zfill(10)
                break

        if not cik:
            return "SEC EDGAR: ไม่พบ CIK"

        # ดึง company facts
        facts = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=headers, timeout=15
        ).json()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def get_annual(concept):
            data = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            annual = [d for d in data if d.get("form") == "10-K"]
            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
            return [(d["end"][:4], round(d["val"]/1e9, 2)) for d in annual[:4]]

        rev  = get_annual("Revenues") or get_annual("RevenueFromContractWithCustomerExcludingAssessedTax")
        ni   = get_annual("NetIncomeLoss")
        debt = get_annual("LongTermDebt")

        lines = [f"=== SEC EDGAR Official (CIK: {cik}) ==="]
        if rev:
            lines.append("Revenue (B$): " + " | ".join([f"{y}: ${v}" for y,v in rev]))
        if ni:
            lines.append("Net Income (B$): " + " | ".join([f"{y}: ${v}" for y,v in ni]))
        if debt:
            lines.append("Long-term Debt (B$): " + " | ".join([f"{y}: ${v}" for y,v in debt]))

        return "\n".join(lines)
    except Exception as e:
        return f"SEC EDGAR: ไม่สามารถดึงข้อมูลได้ ({e})"

# ===== OPTIONS DATA =====
@st.cache_data(ttl=1800)
def get_options_data(ticker: str) -> dict | None:
    """ดึง Put/Call Ratio, IV, Max Pain จาก yfinance options"""
    try:
        hub   = get_ticker_hub(ticker)
        exps  = hub["options_exp"]
        stock = yf.Ticker(ticker)
        if not exps:
            return None

        opt       = stock.option_chain(exps[0])
        calls     = opt.calls
        puts      = opt.puts
        cur_price = stock.fast_info.last_price

        # Put/Call Ratio
        call_vol  = calls["volume"].sum()
        put_vol   = puts["volume"].sum()
        pc_ratio  = round(put_vol / call_vol, 2) if call_vol > 0 else 0

        # Implied Volatility (ATM)
        atm_c = calls[abs(calls["strike"] - cur_price) < cur_price * 0.05]
        atm_p = puts[abs(puts["strike"]  - cur_price) < cur_price * 0.05]
        iv    = pd.concat([atm_c["impliedVolatility"],
                           atm_p["impliedVolatility"]]).mean()
        iv_pct = round(iv * 100, 1) if not pd.isna(iv) else 0

        # Max Pain
        strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        pain = {}
        for s in strikes:
            pain[s] = (calls[calls["strike"] >= s]["openInterest"].sum() +
                       puts[puts["strike"]  <= s]["openInterest"].sum())
        max_pain = min(pain, key=pain.get) if pain else 0

        # Signals
        pc_signal = ("Bearish มาก" if pc_ratio > 1.5 else
                     "กลาง-ลบ"     if pc_ratio > 1.0 else
                     "Bullish"      if pc_ratio < 0.5 else "กลาง")
        iv_signal = ("สูง (ตลาดกังวล)" if iv_pct > 50 else
                     "ต่ำ (ตลาดนิ่ง)"   if iv_pct < 25 else "ปกติ")

        return {
            "pc_ratio":   pc_ratio,
            "pc_signal":  pc_signal,
            "iv":         iv_pct,
            "iv_signal":  iv_signal,
            "max_pain":   round(max_pain, 2),
            "expiration": exps[0],
            "call_vol":   int(call_vol),
            "put_vol":    int(put_vol),
            "summary":    (f"Put/Call Ratio: {pc_ratio} ({pc_signal}) | "
                           f"IV: {iv_pct}% ({iv_signal}) | "
                           f"Max Pain: ${max_pain:.2f} | "
                           f"Expiry: {exps[0]}"),
        }
    except:
        return None


# ===== VALUATION CONTEXT =====
@st.cache_data(ttl=1800)
def get_valuation_context(ticker: str) -> str:
    try:
        hub   = get_ticker_hub(ticker)
        info  = hub["info"]
        if not info:
            return "Valuation Context: ดึงไม่ได้ (rate limit)"
        stock = yf.Ticker(ticker)
        pe    = info.get("trailingPE", 0) or 0
        fwd_pe = info.get("forwardPE", 0) or 0
        pb    = info.get("priceToBook", 0) or 0
        ps    = info.get("priceToSalesTrailing12Months", 0) or 0
        peg   = info.get("pegRatio", 0) or 0
        high_52w  = info.get("fiftyTwoWeekHigh", 0) or 0
        low_52w   = info.get("fiftyTwoWeekLow", 0) or 0
        cur_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        pct_h = round((cur_price - high_52w) / high_52w * 100, 1) if high_52w else 0
        pct_l = round((cur_price - low_52w)  / low_52w  * 100, 1) if low_52w  else 0
        beta  = info.get("beta", 0) or 0
        lines = [
            f"P/E: {pe:.1f}x | Forward P/E: {fwd_pe:.1f}x | P/B: {pb:.1f}x",
            f"P/S: {ps:.1f}x | PEG: {peg:.1f}x | Beta: {beta:.2f}",
            f"52W High: ${high_52w:.2f} ({pct_h:+.1f}%) | 52W Low: ${low_52w:.2f} ({pct_l:+.1f}%)",
        ]
        return "\n".join(lines)
    except:
        return "Valuation Context: ดึงไม่ได้"

# ===== RELATIVE STRENGTH =====
@st.cache_data(ttl=3600)
def get_relative_strength(ticker: str, sector: str) -> str:
    sector_etf = {
        "Technology": "XLK", "Information Technology": "XLK",
        "Healthcare": "XLV", "Financials": "XLF",
        "Energy": "XLE", "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP", "Industrials": "XLI",
        "Materials": "XLB", "Utilities": "XLU",
        "Real Estate": "XLRE", "Communication Services": "XLC",
    }
    etf = sector_etf.get(sector, "SPY")
    results = {}
    for sym, label in [(ticker, "หุ้นนี้"), ("SPY", "S&P 500"), (etf, f"Sector ETF({etf})")]:
        try:
            h = yf.Ticker(sym).history(period="6mo")
            if h.empty:
                continue
            closes = h["Close"].dropna()
            if len(closes) < 5:
                continue
            n = len(closes)
            def safe_ret(a, b):
                try:
                    v = round((a/b - 1)*100, 1)
                    return v if str(v) != "nan" else 0
                except:
                    return 0
            results[label] = {
                "1M": safe_ret(closes.iloc[-1], closes.iloc[-22]) if n>=22 else 0,
                "3M": safe_ret(closes.iloc[-1], closes.iloc[-66]) if n>=66 else 0,
                "6M": safe_ret(closes.iloc[-1], closes.iloc[0]),
            }
        except:
            pass
    if not results:
        return "Relative Strength: ดึงไม่ได้"
    lines = ["Relative Performance:"]
    for label, r in results.items():
        lines.append(f"  {label:20} 1M:{r['1M']:+6.1f}% 3M:{r['3M']:+6.1f}% 6M:{r['6M']:+6.1f}%")
    if "หุ้นนี้" in results and "S&P 500" in results:
        diff = results["หุ้นนี้"]["3M"] - results["S&P 500"]["3M"]
        lines.append(f"vs S&P500(3M): {'Outperform ✅' if diff>0 else 'Underperform ⚠️'} {diff:+.1f}%")
    return "\n".join(lines)

# ===== MANAGEMENT CREDIBILITY =====
@st.cache_data(ttl=1800)
def get_management_credibility(ticker: str) -> str:
    try:
        hub      = get_ticker_hub(ticker)
        earnings = hub["earnings_history"]
        if earnings is None or earnings.empty:
            return "Management: ไม่มีข้อมูล"
        beat = miss = 0
        total_surp = 0
        records = []
        for _, row in earnings.tail(8).iterrows():
            actual   = row.get("epsActual")
            estimate = row.get("epsEstimate")
            if actual is None or estimate is None or estimate == 0:
                continue
            surp = round((actual - estimate)/abs(estimate)*100, 1)
            total_surp += surp
            beat += 1 if surp > 0 else 0
            miss += 1 if surp <= 0 else 0
            records.append(f"  {str(row.name)[:10]}: EPS {actual:.2f} vs คาด {estimate:.2f} ({surp:+.1f}%)")
        total = beat + miss
        if total == 0:
            return "Management: ข้อมูลไม่เพียงพอ"
        beat_rate = round(beat/total*100)
        avg_surp  = round(total_surp/total, 1)
        cred = ("สูง ✅ Beat สม่ำเสมอ" if beat_rate>=75 else
                "กลาง ⚠️ Beat บ้าง Miss บ้าง" if beat_rate>=50 else
                "ต่ำ ❌ Miss บ่อย")
        return (f"Management Credibility: {cred}\n"
                f"Beat Rate: {beat_rate}% ({beat}/{total}) | Avg Surprise: {avg_surp:+.1f}%\n"
                + "\n".join(records))
    except:
        return "Management Credibility: ดึงไม่ได้"

# ===== DCF VALUATION =====
@st.cache_data(ttl=1800)
def calc_simple_dcf(ticker: str) -> str:
    try:
        hub   = get_ticker_hub(ticker)
        info  = hub["info"]
        cf    = hub["cashflow"]
        if not info:
            return "DCF: ดึงข้อมูลไม่ได้ (rate limit)"
        stock = yf.Ticker(ticker)
        if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
            return "DCF: ไม่มีข้อมูล FCF"
        fcf_vals = cf.loc["Free Cash Flow"].dropna()
        if len(fcf_vals) < 2 or fcf_vals.iloc[0] <= 0:
            return f"DCF: FCF ติดลบหรือไม่มีข้อมูล ไม่สามารถคำนวณได้"
        growth = max(min((fcf_vals.iloc[0]/fcf_vals.iloc[-1])**(1/(len(fcf_vals)-1))-1, 0.25), -0.10)
        wacc = 0.10; terminal_g = 0.03
        shares = info.get("sharesOutstanding", 1) or 1
        pv = 0; fcf = fcf_vals.iloc[0]
        for y in range(1, 11):
            fcf *= (1+growth); pv += fcf/(1+wacc)**y
        terminal = fcf*(1+terminal_g)/(wacc-terminal_g)
        pv += terminal/(1+wacc)**10
        intrinsic = round(pv/shares, 2)
        cur = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        margin = round((intrinsic-cur)/cur*100, 1) if cur else 0
        verdict = ("Undervalued น่าสนใจ ✅" if margin>20 else
                   "Fair Value 🟡" if margin>-10 else
                   "Overvalued ระวัง ❌")
        return (f"DCF Intrinsic Value: ${intrinsic:.2f} | ราคาปัจจุบัน: ${cur:.2f}\n"
                f"Margin of Safety: {margin:+.1f}% → {verdict}\n"
                f"(FCF Growth: {growth*100:.1f}% | WACC: 10% | Terminal: 3%)")
    except Exception as e:
        return f"DCF: คำนวณไม่ได้ ({e})"

# ===== EARNINGS TRANSCRIPT (SEC 8-K) =====
@st.cache_data(ttl=86400)
def get_earnings_transcript(ticker: str) -> str:
    try:
        headers = {"User-Agent": "StockAnalyzer research@example.com"}
        tickers_json = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers, timeout=10
        ).json()
        cik = None
        for _, val in tickers_json.items():
            if val.get("ticker","").upper() == ticker.upper():
                cik = str(val["cik_str"]).zfill(10); break
        if not cik:
            return "8-K: ไม่พบ CIK"
        subs = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=headers, timeout=10
        ).json()
        recent = subs.get("filings",{}).get("recent",{})
        forms  = recent.get("form",[])
        dates  = recent.get("filingDate",[])
        accnos = recent.get("accessionNumber",[])
        latest_8k = next(
            ({"date": dates[i], "accno": accnos[i].replace("-","")}
             for i, f in enumerate(forms) if f == "8-K"), None
        )
        if not latest_8k:
            return "8-K: ไม่พบ filing"
        idx = requests.get(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{latest_8k['accno']}/index.json",
            headers=headers, timeout=10
        ).json()
        for doc in idx.get("directory",{}).get("item",[]):
            name = doc.get("name","").lower()
            if any(x in name for x in ["ex99","exhibit99","press","earnings"]):
                url  = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{latest_8k['accno']}/{doc['name']}"
                text = requests.get(url, headers=headers, timeout=10).text
                clean = " ".join(text.split())[:2500]
                return f"[8-K Earnings Release {latest_8k['date']}]\n{clean}"
        return f"พบ 8-K ({latest_8k['date']}) แต่ดึงเนื้อหาไม่ได้"
    except Exception as e:
        return f"8-K: ดึงไม่ได้ ({e})"


# ===== ECONOMIC CALENDAR =====
@st.cache_data(ttl=86400)
def get_economic_calendar() -> dict:
    """FOMC dates + upcoming macro events"""
    fomc_2026 = ["2026-06-18","2026-07-30","2026-09-17","2026-11-05","2026-12-17"]
    today     = datetime.now().date()
    upcoming  = [d for d in fomc_2026 if d >= str(today)]
    next_fomc = upcoming[0] if upcoming else "N/A"
    days_away = (datetime.strptime(next_fomc,"%Y-%m-%d").date()-today).days if next_fomc!="N/A" else 0
    warning   = f"⚠️ FOMC ใน {days_away} วัน — ระวัง volatility" if days_away <= 7 else ""
    # CPI release (approx 2nd week of each month)
    next_month = (today.replace(day=1) + _td(days=32)).replace(day=12)
    return {
        "next_fomc":    next_fomc,
        "days_to_fomc": days_away,
        "fomc_warning": warning,
        "next_cpi_est": str(next_month),
        "summary":      f"FOMC ถัดไป: {next_fomc} ({days_away} วัน) | CPI est.: {next_month}"
    }

# ===== SECTOR ROTATION =====
@st.cache_data(ttl=3600)
def get_sector_rotation() -> str:
    """เปรียบเทียบ performance ทุก sector 3 เดือน"""
    etfs = {
        "Technology":"XLK","Healthcare":"XLV","Financials":"XLF",
        "Energy":"XLE","Consumer Disc":"XLY","Industrials":"XLI",
        "Utilities":"XLU","Materials":"XLB","Real Estate":"XLRE",
        "Consumer Staples":"XLP","Communication":"XLC",
    }
    results = {}
    for sector, etf in etfs.items():
        try:
            h = yf.Ticker(etf).history(period="3mo")
            closes = h["Close"].dropna()
            if len(closes) >= 2:
                ret = round((closes.iloc[-1]/closes.iloc[0]-1)*100,1)
                results[sector] = ret
        except:
            pass
    if not results:
        return "ไม่สามารถดึงข้อมูล sector rotation"
    sorted_r  = sorted(results.items(), key=lambda x:x[1], reverse=True)
    top3      = sorted_r[:3]
    bot3      = sorted_r[-3:]
    lines     = ["=== Sector Rotation (3M) ==="]
    lines    += [f"  ↑ {s}: +{v}%" for s,v in top3]
    lines    += [f"  ↓ {s}: {v}%" for s,v in bot3]
    return "\n".join(lines)

# ===== DCF 3 SCENARIOS =====
@st.cache_data(ttl=86400)
def calculate_dcf_scenarios(ticker: str) -> str:
    """DCF แบบ Bull/Base/Bear scenarios"""
    try:
        stock  = yf.Ticker(ticker)
        info   = stock.info
        cf     = stock.cashflow
        if cf is None or "Free Cash Flow" not in cf.index:
            return "DCF: ไม่มีข้อมูล FCF"
        fcf_vals = cf.loc["Free Cash Flow"].dropna()
        if fcf_vals.iloc[0] <= 0:
            return "DCF: FCF ติดลบ ไม่คำนวณ DCF"
        latest_fcf = fcf_vals.iloc[0]
        shares     = info.get("sharesOutstanding",1) or 1
        cur_price  = info.get("currentPrice") or info.get("regularMarketPrice",0)
        scenarios  = {
            "Bear": {"growth":-0.05,"wacc":0.12,"terminal":0.02},
            "Base": {"growth": 0.10,"wacc":0.10,"terminal":0.03},
            "Bull": {"growth": 0.25,"wacc":0.08,"terminal":0.04},
        }
        lines = ["=== DCF 3 Scenarios ==="]
        for name, p in scenarios.items():
            pv = 0; fcf = latest_fcf
            for y in range(1,11):
                fcf *= (1+p["growth"]); pv += fcf/(1+p["wacc"])**y
            terminal = fcf*(1+p["terminal"])/(p["wacc"]-p["terminal"])
            pv += terminal/(1+p["wacc"])**10
            iv  = round(pv/shares,2)
            margin = round((iv-cur_price)/cur_price*100,1) if cur_price else 0
            emoji = "🟢" if margin>20 else "🔴" if margin<-10 else "🟡"
            lines.append(f"  {emoji} {name}: ${iv:.2f} ({margin:+.1f}% vs ราคา)")
        return "\n".join(lines)
    except Exception as e:
        return f"DCF Scenarios: ไม่ได้ ({e})"


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

# ===== AGENTS (Chain-of-Thought + Evidence Required) =====

def run_agent(prompt, max_tokens=1000):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

def financial_agent(company, financials, quarterly, analyst,
                    sec_data="", valuation_ctx="", mgmt_cred="",
                    dcf_val="", earnings_tx=""):
    return run_agent(f"""คุณเป็น Senior Financial Analyst ระดับ CFA
วิเคราะห์ {company['ticker']} ({company['name']}) โดยคิดทีละขั้นตอน

=== งบการเงินรายปี ===
{financials['income_stmt']}
{financials['balance_sheet']}
{financials['cash_flow']}

=== งบรายไตรมาส ===
{quarterly.get('quarterly_income','')}

=== SEC EDGAR Official ===
{sec_data}

=== Valuation Context ===
{valuation_ctx}

=== Management Credibility ===
{mgmt_cred}

=== DCF Valuation ===
{dcf_val}

=== Earnings Release (8-K) ===
{earnings_tx}

=== Analyst Consensus ===
{analyst}

คิดทีละขั้น:

ขั้น 1 — Revenue Quality
- Growth YoY แต่ละปี (คำนวณจริง) และ momentum เร่งหรือชะลอ
- Gross Margin trend เพราะอะไร
- FCF/Net Income ratio (>80%=ดี, <50%=ระวัง)

ขั้น 2 — งบดุล
- Net Debt = หนี้รวม - เงินสด (คำนวณจริง)
- Cash Runway ถ้า FCF ติดลบ
- Interest Coverage = EBIT ÷ ดอกเบี้ย

ขั้น 3 — Valuation
- P/E ปัจจุบัน vs Forward P/E vs PEG บ่งชี้อะไร
- DCF บอกว่าแพงหรือถูก

ขั้น 4 — Management และ Consensus
- Beat/Miss history บอกอะไรเกี่ยวกับผู้บริหาร
- Analyst consensus สอดคล้องกับงบมั้ย

สรุป:
- จุดแข็ง 3 อย่าง (ตัวเลขทุกข้อ)
- จุดอ่อน 3 อย่าง (ตัวเลขทุกข้อ)
- Confidence: สูง/กลาง/ต่ำ + เหตุผล

ห้ามสรุปโดยไม่มีตัวเลขอ้างอิง
ตอบเป็นภาษาไทย""", 1500)

def macro_agent(company, macro, yield_curve):
    return run_agent(f"""คุณเป็น Chief Macro Economist
วิเคราะห์ผลกระทบ macro ต่อ {company['ticker']} โดยคิดทีละขั้น

=== Macro Indicators ===
{macro}
Yield Curve: {yield_curve}
Sector: {company['sector']} | ธุรกิจ: {company['summary']}

คิดทีละขั้น:

ขั้น 1 — ผลกระทบดอกเบี้ยต่อธุรกิจนี้โดยตรง
- บริษัทนี้มีหนี้มาก/น้อย WACC เปลี่ยนยังไง
- ลูกค้าของธุรกิจนี้ได้รับผลกระทบยังไง

ขั้น 2 — ผลกระทบเงินเฟ้อ
- ต้นทุนหลักคืออะไร เงินเฟ้อกดดันแค่ไหน
- Pricing power มีหรือไม่

ขั้น 3 — GDP/Unemployment/PMI/Consumer Sentiment บอกอะไร
- เศรษฐกิจกำลังขยายหรือหดตัว
- ส่งผลต่อ demand ของธุรกิจนี้อย่างไร

ขั้น 4 — Yield Curve
- Inverted = ระวัง recession หรือ Normal = ขยายตัว
- กระทบ sector นี้อย่างไร

สรุป:
- Macro เอื้อ/เป็นกลาง/เป็นอุปสรรค + เหตุผล
- ความเสี่ยง macro 2 อย่างที่ใหญ่ที่สุด
- Confidence: สูง/กลาง/ต่ำ

ตอบเป็นภาษาไทย อ้างตัวเลข macro จริง""", 900)

def news_agent(company, news):
    return run_agent(f"""คุณเป็น Senior News Analyst และ Behavioral Finance Expert
วิเคราะห์ข่าว {company['ticker']} ({company['name']})

=== ข่าว ===
{news}

คิดทีละขั้น:

ขั้น 1 — แยกประเภทข่าว
- ข่าวกระทบ fundamental จริง (earnings, contract, product launch)
- ข่าวกระทบ sentiment เท่านั้น (opinion, forecast, analyst note)
- ข่าว noise (ไม่มีนัยสำคัญ)

ขั้น 2 — Sentiment
- Bullish/Bearish/Neutral + น้ำหนัก (เช่น Bullish 70%)
- ข่าวไหนมีผลมากที่สุดและทำไม

ขั้น 3 — Catalyst และ Risk
- Catalyst ที่อาจทำให้ราคาพุ่ง
- ข่าวเสี่ยงที่อาจกดราคา
- Event สำคัญที่ต้องจับตา

สรุป + Confidence: สูง/กลาง/ต่ำ
ตอบเป็นภาษาไทย""", 800)

def technical_agent(company, price_summary, indicators,
                    options_summary="", relative_strength=""):
    opts = f"\n=== Options Data ===\n{options_summary}" if options_summary else ""
    rs   = f"\n=== Relative Strength ===\n{relative_strength}" if relative_strength else ""
    return run_agent(f"""คุณเป็น Senior Technical Analyst
วิเคราะห์ {company['ticker']} โดยคิดทีละขั้น

=== Price & Indicators ===
{price_summary}
{indicators}
{opts}
{rs}

คิดทีละขั้น:

ขั้น 1 — โครงสร้างราคา
- ราคาอยู่ที่ % ไหนของ 52W range
- เทียบ MA20/MA50 (เหนือ/ต่ำกว่า กี่%)
- Trend หลัก: Uptrend/Downtrend/Sideways

ขั้น 2 — Momentum Indicators
- RSI: Overbought(>70)/Oversold(<30)/Neutral + นัยยะ
- MACD: Bullish/Bearish + มี divergence มั้ย
- Stochastic: สัญญาณเข้า/ออก
- ATR: ควรตั้ง Stop ห่างกี่ % จากราคา

ขั้น 3 — Options Sentiment
- Put/Call Ratio บ่งชี้อะไร
- Max Pain vs ราคาปัจจุบัน ต่างกันมั้ย
- IV สูง/ต่ำ บอกอะไรเกี่ยวกับ expected move

ขั้น 4 — Relative Performance
- Outperform หรือ Underperform vs S&P500 และ Sector
- บ่งชี้อะไรเกี่ยวกับ momentum

สรุป:
- แนวรับหลัก 2 ระดับ (ราคาจริง)
- แนวต้านหลัก 2 ระดับ (ราคาจริง)
- Scenario ถ้าขึ้น vs ถ้าลง
- Confidence: สูง/กลาง/ต่ำ

ตอบเป็นภาษาไทย อ้างราคาจริงทุกจุด""", 1000)

def build_orchestrator_prompt(company, fin, mac, geo, insider, news, tech,
                               eco_cal="", sector_rot="", dcf_scenarios="",
                               conditional_summaries="") -> str:
    """สร้าง prompt สำหรับ Orchestrator"""
    extra = ""
    if eco_cal:
        extra += f"\n=== Economic Calendar ===\n{eco_cal}"
    if sector_rot:
        extra += f"\n=== Sector Rotation ===\n{sector_rot}"
    if dcf_scenarios:
        extra += f"\n=== DCF 3 Scenarios ===\n{dcf_scenarios}"
    if conditional_summaries:
        extra += f"\n=== Special Analysis (Conditional Agents) ===\n{conditional_summaries}"

    return f"""คุณเป็น Chief Investment Officer
รวมผลจากทีมผู้เชี่ยวชาญและสรุปภาพรวมการลงทุน

หุ้น: {company['ticker']} ({company['name']})
ราคา: ${company['price']:.2f} | Market Cap: ${company['market_cap_b']:.1f}B
{extra}

[Financial Analyst]: {fin}
[Macro Economist]: {mac}
[Geopolitical Analyst]: {geo}
[Insider & Market Structure]: {insider}
[News Analyst]: {news}
[Technical Analyst]: {tech}

คิดทีละขั้น:

ขั้น 1 — Bull Case (อ้างหลักฐานจาก agents)
ขั้น 2 — Bear Case (อ้างหลักฐานจาก agents)
ขั้น 3 — จุดเข้าซื้อ: Zone A (Aggressive), Zone B (Optimal), Zone C (Conservative)
ขั้น 4 — Stop Loss และ Target 1, Target 2
ขั้น 5 — สรุป: ความเสี่ยง, กลยุทธ์, Confidence

ตอบเป็นภาษาไทย ละเอียด มีตัวเลขชัดเจน"""

def stream_orchestrator(prompt: str, placeholder):
    """Streaming version ของ Orchestrator — แสดง token by token"""
    full_text = ""
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=3500,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as e:
        full_text = run_agent(prompt, 3500)
        placeholder.markdown(full_text)
    return full_text

def orchestrator_agent_with_tools(company, fin, mac, geo, insider, news, tech,
                                   eco_cal="", sector_rot="", dcf_scenarios="",
                                   conditional_summaries="") -> str:
    """Orchestrator พร้อม Tool Calling สำหรับ follow-up queries"""
    tools = [
        {
            "name": "search_recent_news",
            "description": "ค้นหาข่าวล่าสุดเพิ่มเติมเมื่อต้องการข้อมูลเพิ่ม",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "คำค้นหาข่าว"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_economic_calendar",
            "description": "ดึงข้อมูล upcoming macro events เช่น FOMC, CPI",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

    prompt   = build_orchestrator_prompt(company, fin, mac, geo, insider, news, tech,
                                          eco_cal, sector_rot, dcf_scenarios,
                                          conditional_summaries)
    messages = [{"role": "user", "content": prompt}]
    max_tool_loops = 3

    for _ in range(max_tool_loops):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3500,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search_recent_news":
                    result = search_news_manual(block.input.get("query",""))
                    result_text = "\n".join([r.get("title","") for r in result[:5]])
                elif block.name == "get_economic_calendar":
                    cal = get_economic_calendar()
                    result_text = cal.get("summary","")
                else:
                    result_text = "ไม่พบ tool"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # คืน text สุดท้าย
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "Orchestrator error"

def orchestrator_agent(company, fin, mac, geo, insider, news, tech):
    """CIO Orchestrator แบบ Self-Reflection 2 รอบ"""

    initial_prompt = f"""คุณเป็น Chief Investment Officer
รวมผลจากทีมผู้เชี่ยวชาญและสรุปภาพรวมการลงทุน

หุ้น: {company['ticker']} ({company['name']})
ราคา: ${company['price']:.2f} | Market Cap: ${company['market_cap_b']:.1f}B

[Financial Analyst]: {fin}
[Macro Economist]: {mac}
[Geopolitical Analyst]: {geo}
[Insider & Market Structure]: {insider}
[News Analyst]: {news}
[Technical Analyst]: {tech}

คิดทีละขั้น:

ขั้น 1 — Bull Case
- เหตุผล 3 อย่างที่น่าลงทุน (อ้างหลักฐานจาก agent)

ขั้น 2 — Bear Case
- ความเสี่ยง 3 อย่างที่ต้องระวัง (อ้างหลักฐานจาก agent)

ขั้น 3 — จุดเข้าซื้อ
- Zone A (Aggressive): ราคา + เหตุผล
- Zone B (Optimal): ราคา + เหตุผล
- Zone C (Conservative): ราคา + เหตุผล

ขั้น 4 — Stop Loss และ Target
- Stop Loss: ราคา + เหตุผลจาก technical
- Target 1: ราคา + เหตุผล
- Target 2: ราคา + เหตุผล

ขั้น 5 — สรุป
- ระดับความเสี่ยงรวม: สูง/กลาง/ต่ำ + เหตุผล
- กลยุทธ์ที่แนะนำ: DCA/รอ pullback/เข้าทันที/หลีกเลี่ยง
- Confidence โดยรวม: สูง/กลาง/ต่ำ

ตอบเป็นภาษาไทย ละเอียด มีตัวเลขชัดเจนทุกจุด"""

    initial = run_agent(initial_prompt, max_tokens=3000)

    # Self-Reflection
    reflection = run_agent(f"""ตรวจสอบการวิเคราะห์นี้อย่างวิจารณ์:

{initial}

ตรวจหา 4 อย่าง:
1. ข้อสรุปที่ขัดแย้งกันระหว่าง agents (เช่น Financial ดีแต่สรุปว่าไม่น่าลงทุน)
2. ความมั่นใจเกินจริงโดยไม่มีหลักฐานพอ
3. ข้อมูลสำคัญจาก agents ที่ถูกมองข้าม
4. จุดเข้า/ออกที่ไม่มีเหตุผลรองรับชัดเจน

ถ้าพบปัญหา → ระบุว่าต้องแก้ตรงไหน
ถ้าไม่พบ → ตอบว่า "การวิเคราะห์สมเหตุสมผล"
ตอบสั้นๆ เป็นภาษาไทย""", 400)

    needs_revision = any(w in reflection for w in [
        "ขัดแย้ง","ปัญหา","ไม่มีหลักฐาน","มองข้าม","ไม่สมเหตุสมผล","ควรแก้"
    ])

    if needs_revision:
        final = run_agent(f"""แก้ไขการวิเคราะห์โดยคำนึงถึงข้อบกพร่องนี้:

การวิเคราะห์เดิม:
{initial}

ข้อบกพร่องที่พบ:
{reflection}

เขียนใหม่โดยแก้ไขจุดที่มีปัญหา
ตอบเป็นภาษาไทย ละเอียด มีตัวเลขชัดเจน""", 3000)
        return f"{final}\n\n---\n[Self-Reflection: ตรวจพบและแก้ไขปัญหา ✅]\n{reflection}"
    else:
        return f"{initial}\n\n---\n[Self-Reflection: ผ่านการตรวจสอบ ✅]"


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


# ===== CONDITIONAL ROUTING =====
def check_conditional_triggers(results: dict, indicators_raw,
                                insider_data: dict, macro_str: str,
                                yield_curve: str) -> list:
    """ตรวจ conditions แล้วคืน list ของ agents ที่ต้องรัน"""
    triggers = []

    # FCF ติดลบ → Distressed
    fin = results.get("financial","").lower()
    if "fcf ติดลบ" in fin or "cash runway" in fin or "burn rate" in fin:
        triggers.append("distressed")

    # Short Interest > 20% → Squeeze Risk
    if insider_data:
        si = insider_data.get("short_pct_float", 0) or 0
        if si >= 20:
            triggers.append("squeeze")

    # Yield curve inverted → Recession Risk
    if "inverted" in yield_curve.lower() or "ระวัง recession" in yield_curve:
        triggers.append("recession")

    # RSI < 25 → Oversold
    tech = results.get("technical","").lower()
    if "rsi" in tech:
        import re as _re2
        m = _re2.search(r"rsi[^0-9]*([0-9]+\.?[0-9]*)", tech)
        if m and float(m.group(1)) < 25:
            triggers.append("oversold")

    # VIX > 30 → Systemic Risk
    geo_txt = results.get("geopolitical","").lower()
    if "vix" in geo_txt:
        import re as _re3
        m2 = _re3.search(r"vix[^0-9]*([0-9]+\.?[0-9]*)", geo_txt)
        if m2 and float(m2.group(1)) > 30:
            triggers.append("systemic")

    return list(set(triggers))

def distressed_agent(company, fin_result: str, eco_cal: str) -> str:
    return run_agent(f"""คุณเป็น Distressed Asset Analyst
{company['ticker']} มี FCF ติดลบ — วิเคราะห์เชิงลึก

{fin_result}
Economic Calendar: {eco_cal}

วิเคราะห์:
1. Cash Runway เหลือกี่เดือน (คำนวณจากเงินสด ÷ burn rate)
2. แหล่งเงินทุนทางเลือก (debt, equity raise, asset sale)
3. Covenant risk — เสี่ยงผิดเงื่อนไขหนี้มั้ย
4. Turnaround probability และ timeline
5. สรุป: ถือ/ขาย/หลีกเลี่ยง

ตอบเป็นภาษาไทย มีตัวเลขชัดเจน""", 800)

def short_squeeze_agent(company, insider_data: dict, tech_result: str) -> str:
    si  = insider_data.get("short_pct_float",0) if insider_data else 0
    dtc = insider_data.get("short_ratio",0) if insider_data else 0
    return run_agent(f"""คุณเป็น Market Structure Analyst เชี่ยวชาญ Short Squeeze
{company['ticker']} มี Short Interest {si}% (Days to Cover: {dtc})

Technical Context: {tech_result}

วิเคราะห์:
1. Short Squeeze Probability (สูง/กลาง/ต่ำ) + เหตุผล
2. Catalyst ที่จะจุด squeeze (earnings, news, sector rotation)
3. ถ้า squeeze เกิด ราคาอาจวิ่งถึงระดับไหน
4. ความเสี่ยงสำหรับคนที่ถือ long ท่ามกลาง short interest สูง
5. กลยุทธ์: เข้าซื้อก่อน squeeze หรือรอ

ตอบเป็นภาษาไทย มีตัวเลขอ้างอิง""", 700)

def recession_risk_agent(company, mac_result: str, yield_curve: str,
                          sector_rot: str) -> str:
    return run_agent(f"""คุณเป็น Macro Strategist เชี่ยวชาญ Recession Analysis
Yield Curve: {yield_curve} — บ่งชี้ recession risk

Macro Context: {mac_result}
Sector Rotation: {sector_rot}

วิเคราะห์สำหรับ {company['ticker']} ({company['sector']}):
1. Yield curve inversion บ่งบอก recession ใน timeline กี่เดือน (historical avg)
2. Sector {company['sector']} ได้รับผลกระทบจาก recession ยังไง
3. บริษัทนี้มี defensive characteristics มั้ย (pricing power, recurring revenue)
4. Sector rotation: capital กำลังไหลไป/ออกจาก sector นี้หรือไม่
5. กลยุทธ์รับมือ recession สำหรับหุ้นตัวนี้

ตอบเป็นภาษาไทย อ้างอิง historical data""", 800)

def systemic_risk_agent(company, geo_result: str, eco_cal: str) -> str:
    return run_agent(f"""คุณเป็น Systemic Risk Analyst
VIX สูงกว่า 30 — ตลาดอยู่ใน Risk-off Mode

Geopolitical Context: {geo_result}
Economic Calendar: {eco_cal}

วิเคราะห์:
1. VIX > 30 หมายความว่าอะไรต่อ positioning ในตลาด
2. Correlation ของ {company['ticker']} กับตลาดในช่วง risk-off (Beta)
3. Safe haven flows จะกระทบ sector นี้อย่างไร
4. FOMC ใกล้มาหรือไม่ — จะซ้ำเติมหรือช่วยบรรเทา
5. กลยุทธ์: hedge, reduce position, หรือ hold

ตอบเป็นภาษาไทย กระชับ""", 700)

def oversold_agent(company, tech_result: str, valuation_ctx: str,
                   mgmt_cred: str) -> str:
    return run_agent(f"""คุณเป็น Contrarian Investment Analyst
{company['ticker']} มี RSI < 25 — Oversold Extreme

Technical: {tech_result}
Valuation: {valuation_ctx}
Management Track Record: {mgmt_cred}

วิเคราะห์:
1. Oversold นี้เป็น genuine value หรือ value trap
2. Fundamental ยังดีอยู่มั้ย — ราคาลงเพราะอะไร
3. Insider ซื้อหรือขายช่วงราคาลง (management confidence)
4. Historical — ครั้งก่อนที่ RSI ต่ำระดับนี้ราคาทำอะไร
5. Entry zone ที่เหมาะสม และ Stop Loss

ตอบเป็นภาษาไทย มีตัวเลขชัดเจน""", 700)


def chat_agent(ticker, company, fin_result, mac_result, geo_result, insider_result, news_result, tech_result, final, messages, earnings_date="N/A"):
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

[Insider & Short Interest]
{insider_result}

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

# Portfolio shortcut ใน title bar
if st.button("My Portfolio", key="port_shortcut"):
    st.session_state["show_portfolio"] = True

# ===== PORTFOLIO PAGE (Dime Style) =====
if st.session_state.get("show_portfolio"):

    if st.button("← กลับ"):
        st.session_state["show_portfolio"] = False
        st.rerun()

    rate      = get_usd_thb_rate()
    positions = portfolio_load_positions()

    if positions:
        tickers_tuple = tuple(set(p["ticker"] for p in positions))
        prices        = portfolio_get_current_prices(tickers_tuple)
        summary       = portfolio_calc_summary(positions, prices)

        # คำนวณ holding list พร้อม badge colors
        holding_list = []
        for i, p in enumerate(positions):
            t        = p["ticker"]
            cur      = prices.get(t, {})
            cur_px   = cur.get("price", p["entry_price"])
            prev_px  = cur.get("prev", cur_px)
            value    = round(p["shares"] * cur_px, 2)
            cost     = round(p["shares"] * p["entry_price"], 2)
            pl       = round(value - cost, 2)
            pl_pct   = round(pl / cost * 100, 2) if cost else 0
            day_chg  = round((cur_px - prev_px) / prev_px * 100, 2) if prev_px else 0
            port_pct = round(value / summary["total_value"] * 100, 1) if summary["total_value"] else 0
            bg, tc   = BADGE_COLORS[i % len(BADGE_COLORS)]
            holding_list.append({**p, "value": value, "cost": cost, "pl": pl,
                                  "pl_pct": pl_pct, "day_chg": day_chg,
                                  "port_pct": port_pct, "cur_price": cur_px,
                                  "bg": bg, "tc": tc})

        holding_list.sort(key=lambda x: x["value"], reverse=True)

        # ===== HEADER CARD (สไตล์ Dime) =====
        pl_pos   = summary["unrealized_pl"] >= 0
        day_pos  = summary["day_change"] >= 0
        pl_sign  = "↑" if pl_pos  else "↓"
        day_sign = "↑" if day_pos else "↓"
        pl_css   = "var(--color-text-success)" if pl_pos  else "var(--color-text-danger)"
        day_css  = "var(--color-text-success)" if day_pos else "var(--color-text-danger)"

        # แถบสีสัดส่วน portfolio
        bar_segs = "".join([
            f'<div style="flex:{h["port_pct"]};height:100%;background:{h["bg"]};'
            f'border:1px solid {h["tc"]};border-radius:2px"></div>'
            for h in holding_list
        ])

        st.markdown(f"""
<div style="background:var(--color-background-secondary);border-radius:14px;padding:20px 22px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:2px;">
    <div style="font-size:12px;color:var(--color-text-tertiary);">มูลค่าพอร์ตรวม</div>
    <div style="font-size:11px;color:var(--color-text-tertiary);">1 USD = {rate:.2f} ฿</div>
  </div>
  <div style="font-size:30px;font-weight:500;color:var(--color-text-primary);line-height:1.2;">${summary['total_value']:,.2f} USD</div>
  <div style="font-size:13px;color:var(--color-text-secondary);margin-top:2px;">≈ {int(summary['total_value']*rate):,} บาท</div>
  <div style="display:flex;gap:2px;height:6px;margin:14px 0 14px;border-radius:3px;overflow:hidden;">{bar_segs}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0;">
    <div style="padding-right:12px;">
      <div style="font-size:11px;color:var(--color-text-tertiary);">เปลี่ยนวันนี้</div>
      <div style="font-size:14px;font-weight:500;color:{day_css};">{day_sign}{abs(summary['day_change_pct']):.2f}%</div>
    </div>
    <div style="padding:0 12px;border-left:0.5px solid var(--color-border-tertiary);">
      <div style="font-size:11px;color:var(--color-text-tertiary);">กำไร/ขาดทุน</div>
      <div style="font-size:14px;font-weight:500;color:{pl_css};">{pl_sign}{abs(summary['unrealized_pct']):.2f}%</div>
      <div style="font-size:11px;color:{pl_css};">{pl_sign}${abs(summary['unrealized_pl']):,.2f}</div>
    </div>
    <div style="padding:0 12px;border-left:0.5px solid var(--color-border-tertiary);">
      <div style="font-size:11px;color:var(--color-text-tertiary);">ต้นทุน</div>
      <div style="font-size:14px;font-weight:500;color:var(--color-text-primary);">${summary['total_cost']:,.2f}</div>
    </div>
    <div style="padding-left:12px;border-left:0.5px solid var(--color-border-tertiary);">
      <div style="font-size:11px;color:var(--color-text-tertiary);">จำนวนหุ้น</div>
      <div style="font-size:14px;font-weight:500;color:var(--color-text-primary);">{len(positions)} ตัว</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ===== HOLDINGS LIST (สไตล์ Dime) =====
        st.caption("HOLDINGS · เรียงตาม Holding Value")

        for h in holding_list:
            pl_css2  = "var(--color-text-success)" if h["pl"] >= 0 else "var(--color-text-danger)"
            pl_arrow = "↑" if h["pl"] >= 0 else "↓"

            # Card หลัก
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:14px 0;border-bottom:0.5px solid var(--color-border-tertiary);">
  <div style="width:44px;height:44px;border-radius:10px;background:{h['bg']};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:500;color:{h['tc']};flex-shrink:0;">{h['ticker'][:4]}</div>
  <div style="flex:1;min-width:0;">
    <div style="font-size:14px;font-weight:500;color:var(--color-text-primary);">{h['ticker']}</div>
    <div style="font-size:11px;color:var(--color-text-tertiary);">{h['port_pct']}% · {h['shares']} หุ้น · เข้า ${h['entry_price']:.2f}</div>
  </div>
  <div style="text-align:right;flex-shrink:0;">
    <div style="font-size:15px;font-weight:500;color:var(--color-text-primary);">${h['value']:,.2f}</div>
    <div style="font-size:11px;color:var(--color-text-tertiary);">≈ {int(h['value']*rate):,} ฿</div>
    <div style="font-size:13px;font-weight:500;color:{pl_css2};">{pl_arrow} {abs(h['pl_pct']):.2f}% (${h['pl']:+,.2f})</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ปุ่ม action
            b1, b2, b3 = st.columns([3, 3, 1])
            if b1.button(f"⚡ Quick View", key=f"qv_{h['id']}"):
                st.session_state["show_portfolio"] = False
                st.session_state["load_ticker"]    = h["ticker"]
                st.rerun()
            if b2.button(f"🤖 Analyze", key=f"an_{h['id']}"):
                st.session_state["show_portfolio"] = False
                st.session_state["load_ticker"]    = h["ticker"]
                st.rerun()
            if b3.button("🗑", key=f"del_{h['id']}"):
                portfolio_delete_position(h["id"])
                st.rerun()

    else:
        st.markdown("""
<div style="text-align:center;padding:40px 20px;color:var(--color-text-tertiary);">
  <div style="font-size:32px;margin-bottom:8px;">📊</div>
  <div style="font-size:14px;">ยังไม่มีหุ้นใน portfolio</div>
  <div style="font-size:12px;margin-top:4px;">เพิ่มหุ้นด้านล่างได้เลย</div>
</div>
""", unsafe_allow_html=True)

    # ===== NEWS SECTION =====
    if positions:
        st.divider()
        st.subheader("📰 ข่าวพอร์ต")
        all_tickers = tuple(set(p["ticker"] for p in positions))

        # Refresh button
        ref_col, _ = st.columns([1, 5])
        if ref_col.button("🔄 โหลดข่าวใหม่", key="refresh_news"):
            st.cache_data.clear()
            st.rerun()

        # Filter buttons
        filter_cols = st.columns(len(all_tickers) + 1)
        if filter_cols[0].button("All", type="primary" if st.session_state["news_filter"]=="All" else "secondary", use_container_width=True):
            st.session_state["news_filter"] = "All"
            st.rerun()
        for i, t in enumerate(all_tickers):
            btn_type = "primary" if st.session_state["news_filter"] == t else "secondary"
            if filter_cols[i+1].button(t, type=btn_type, use_container_width=True):
                st.session_state["news_filter"] = t
                st.rerun()

        # Auto Feed
        with st.spinner("กำลังดึงข่าวล่าสุด..."):
            all_news = get_portfolio_news(all_tickers, max_per=5)

        filtered = (all_news if st.session_state["news_filter"] == "All"
                    else [n for n in all_news if n["ticker"] == st.session_state["news_filter"]])

        if filtered:
            for n in filtered:
                bg, tc = next(
                    (BADGE_COLORS[i % len(BADGE_COLORS)] for i, p in enumerate(positions) if p["ticker"] == n.get("ticker","")),
                    ("#E6F1FB","#185FA5")
                )
                pub = f"· {n['published']}" if n.get("published") else ""
                st.markdown(f"""
<div style="display:flex;gap:12px;padding:12px 0;border-bottom:0.5px solid var(--color-border-tertiary);align-items:flex-start;">
  <div style="width:36px;height:36px;border-radius:8px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:{tc};flex-shrink:0;">{n.get('ticker','')[:4]}</div>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
      <span style="font-size:11px;font-weight:500;color:{n['sent_color']};">{n['sentiment']}</span>
      <span style="font-size:11px;color:var(--color-text-tertiary);">{pub}</span>
    </div>
    <div style="font-size:13px;font-weight:500;color:var(--color-text-primary);line-height:1.4;">{n['title']}</div>
    <div style="font-size:12px;color:var(--color-text-secondary);margin-top:3px;">{n.get('content','')}</div>
  </div>
</div>
""", unsafe_allow_html=True)
                if n.get("url"):
                    st.link_button("อ่านต่อ →", n["url"])
        else:
            st.caption("ไม่พบข่าว")

        # Manual Search
        st.divider()
        st.subheader("🔍 ค้นข่าวเพิ่มเติม")
        sc1, sc2 = st.columns([4, 1])
        search_q = sc1.text_input("", placeholder="เช่น quantum computing tariff, IONQ government contract", label_visibility="collapsed")
        do_search = sc2.button("ค้นหา", use_container_width=True)

        if do_search and search_q:
            st.session_state["news_search_query"] = search_q
        if st.session_state.get("news_search_query"):
            with st.spinner("กำลังค้นหา..."):
                manual_results = search_news_manual(st.session_state["news_search_query"])
            if manual_results:
                for r in manual_results:
                    st.markdown(f"""
<div style="padding:10px 0;border-bottom:0.5px solid var(--color-border-tertiary);">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
    <span style="font-size:11px;font-weight:500;color:{r['sent_color']};">{r['sentiment']}</span>
    <span style="font-size:11px;color:var(--color-text-tertiary);">{r.get('published','')}</span>
  </div>
  <div style="font-size:13px;font-weight:500;color:var(--color-text-primary);">{r['title']}</div>
  <div style="font-size:12px;color:var(--color-text-secondary);margin-top:2px;">{r.get('content','')}</div>
</div>
""", unsafe_allow_html=True)
                    if r.get("url"):
                        st.link_button("อ่านต่อ →", r["url"])
            else:
                st.info("ไม่พบผลลัพธ์ หรือยังไม่ได้ตั้งค่า Tavily API Key")

        st.divider()

    # ===== ADD POSITION FORM =====
    st.subheader("เพิ่มหุ้น")
    with st.form("add_position_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        f_ticker  = fc1.text_input("Ticker", placeholder="IONQ").upper().strip()
        f_shares  = fc2.number_input("จำนวนหุ้น", min_value=0.01, step=1.0, value=100.0)
        fc3, fc4  = st.columns(2)
        f_price   = fc3.number_input("ราคาที่เข้า ($)", min_value=0.01, step=0.01, value=0.01)
        f_date    = fc4.date_input("วันที่เข้า", value=datetime.now().date())
        f_notes   = st.text_input("หมายเหตุ (ไม่บังคับ)", placeholder="เช่น DCA รอบที่ 1")

        # ถ้าไม่ใส่ราคา — ดึงราคาปัจจุบันให้อัตโนมัติ
        use_market = st.checkbox("ใช้ราคาตลาดปัจจุบัน", value=False)
        submitted  = st.form_submit_button("เพิ่มลง Portfolio", use_container_width=True)

        if submitted and f_ticker:
            if use_market or f_price <= 0.01:
                try:
                    f_price = yf.Ticker(f_ticker).fast_info.last_price
                except:
                    st.error("ดึงราคาตลาดไม่ได้ กรุณาใส่ราคาเอง")
                    f_price = 0
            if f_price > 0 and f_shares > 0:
                try:
                    cname = yf.Ticker(f_ticker).info.get("shortName", f_ticker)
                except:
                    cname = f_ticker
                ok = portfolio_add_position(f_ticker, cname, f_shares, f_price,
                                             str(f_date), f_notes)
                if ok:
                    portfolio_add_transaction(f_ticker, "BUY", f_shares, f_price)
                    st.success(f"เพิ่ม {f_ticker} × {f_shares} @ ${f_price:.2f} แล้ว")
                    st.rerun()

    # ===== TRANSACTION HISTORY =====
    with st.expander("ประวัติการซื้อขาย"):
        txns = portfolio_load_transactions()
        if txns:
            for tx in txns:
                sign = "BUY" if tx["action"] == "BUY" else "SELL"
                col  = "var(--color-text-success)" if tx["action"] == "BUY" else "var(--color-text-danger)"
                st.markdown(
                    f'<span style="color:{col};font-size:12px;">{sign}</span> '
                    f'<span style="font-size:13px;">{tx["ticker"]} × {tx["shares"]} @ ${tx["price"]:.2f} '
                    f'= ${tx["amount"]:,.2f}</span> '
                    f'<span style="font-size:11px;color:var(--color-text-tertiary);">{str(tx.get("created_at",""))[:10]}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("ยังไม่มีประวัติ")

    st.stop()

if "load_ticker" in st.session_state:
    ticker_input = st.session_state.pop("load_ticker")
else:
    ticker_input = st.text_input("ใส่ ticker", placeholder="เช่น BE, NVDA, TSLA").upper().strip()

col_q, col_a = st.columns([1, 1])
btn_quick    = col_q.button("⚡ Quick View (ฟรี)", use_container_width=True)
btn_analyze  = col_a.button("🤖 Full Analyze (~$0.09)", type="primary", use_container_width=True)

# ===== QUICK VIEW MODE =====
if btn_quick and ticker_input:
    with st.spinner("กำลังดึงข้อมูล..."):
        import time as _time
        company = get_company_info(ticker_input)
        if not company:
            st.error(f"ไม่พบข้อมูล {ticker_input}")
            st.stop()
        hist, fin, bs, cf = get_chart_data(ticker_input)
        price_summary     = get_price_summary(hist)
        indicators_raw    = get_technical_indicators(hist)
        _time.sleep(0.3)   # หลีกเลี่ยง rate limit
        options_data      = get_options_data(ticker_input)
        valuation_ctx     = get_valuation_context(ticker_input)
        _time.sleep(0.3)
        relative_str      = get_relative_strength(ticker_input, company.get("sector","Technology"))
        mgmt_cred         = get_management_credibility(ticker_input)
        _time.sleep(0.3)
        dcf_val           = calc_simple_dcf(ticker_input)
        macro             = get_macro_data()
        geo_ind           = get_geopolitical_indicators()
        insider_short     = get_insider_short_data(ticker_input)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ราคา",       f"${company['price']:.2f}")
    c2.metric("Market Cap", f"${company['market_cap_b']:.1f}B")
    c3.metric("Sector",     company['sector'])
    c4.metric("Industry",   company['industry'])

    st.divider()

    # Charts
    st.subheader("กราฟ")
    ALL_CHARTS_QV = ["ราคา + MA","Bollinger Bands","Trendline",
                     "Volume","RSI","Revenue","Gross Margin","Free Cash Flow","Debt vs Cash"]
    sel_qv = st.multiselect("เลือกกราฟ (สูงสุด 2)", ALL_CHARTS_QV,
                             default=["ราคา + MA","Revenue"], max_selections=2, key="qv_charts")
    if len(sel_qv) == 2:
        gc1, gc2 = st.columns(2)
        with gc1: draw_chart(sel_qv[0], ticker_input, hist, fin, bs, cf)
        with gc2: draw_chart(sel_qv[1], ticker_input, hist, fin, bs, cf)
    elif len(sel_qv) == 1:
        draw_chart(sel_qv[0], ticker_input, hist, fin, bs, cf)

    st.divider()

    # Quick data grid
    st.subheader("ข้อมูลสำคัญ (ไม่ต้องใช้ AI)")

    qv1, qv2 = st.columns(2)
    with qv1:
        with st.expander("Valuation & DCF", expanded=True):
            st.text(valuation_ctx)
            st.text(dcf_val)
        with st.expander("Technical Snapshot", expanded=True):
            # แก้ $nan — ใช้ fast_info เป็น fallback
            safe_summary = price_summary.replace("$nan", "$N/A").replace("nan", "N/A")
            st.text(safe_summary)
            if indicators_raw:
                ind_display = str(indicators_raw).replace("nan", "N/A").replace("$nan", "$N/A")
                st.text(ind_display)
        with st.expander("Management Credibility", expanded=True):
            st.text(mgmt_cred)

    with qv2:
        with st.expander("Options Data", expanded=True):
            if options_data:
                st.text(options_data["summary"])
            else:
                st.text("ไม่มีข้อมูล options")
        with st.expander("Macro & Geopolitical", expanded=True):
            st.text(macro)
            geo_str = "\n".join([f"{k}: {v}" for k,v in geo_ind.items()]) if geo_ind else "N/A"
            st.text(geo_str)
        with st.expander("Relative Performance", expanded=True):
            st.text(relative_str.replace("nan", "N/A"))
        if insider_short:
            with st.expander("Short Interest", expanded=True):
                st.text(
                    f"Short % of Float: {insider_short.get('short_pct_float',0)}% | "
                    f"Days to Cover: {insider_short.get('short_ratio',0)}"
                )

    st.divider()
    st.info("กด **🤖 Full Analyze** เพื่อให้ AI วิเคราะห์เชิงลึกพร้อมจุดเข้า/ออก (~$0.09)")

if btn_analyze and ticker_input:

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
        earnings_date  = "N/A"
        insider_result = cached.get("insider_result", "ไม่มีข้อมูล insider")
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
            news              = get_news(ticker_input, company.get("name",""))
            macro             = get_macro_data()
            yield_curve       = get_yield_curve()
            geo_indicators    = get_geopolitical_indicators()
            insider_data      = get_insider_short_data(ticker_input)
            sec_data          = get_sec_financials(ticker_input)
            options_data      = get_options_data(ticker_input)
            options_summary   = options_data["summary"] if options_data else ""
            valuation_ctx     = get_valuation_context(ticker_input)
            relative_strength = get_relative_strength(ticker_input, company.get("sector","Technology"))
            mgmt_cred         = get_management_credibility(ticker_input)
            dcf_val           = calc_simple_dcf(ticker_input)
            earnings_tx       = get_earnings_transcript(ticker_input)
            eco_cal           = get_economic_calendar()
            sector_rot        = get_sector_rotation()
            dcf_scenarios_str = calculate_dcf_scenarios(ticker_input)

        # ===== PARALLEL AGENTS + CHECKPOINT =====
        agent_tasks = {
            "financial":    lambda: financial_agent(
                company, financials, quarterly, analyst,
                sec_data, valuation_ctx, mgmt_cred, dcf_val, earnings_tx
            ),
            "macro":        lambda: macro_agent(company, macro, yield_curve),
            "news":         lambda: news_agent(company, news),
            "technical":    lambda: technical_agent(
                company, price_summary, indicators,
                options_summary, relative_strength
            ),
            "geopolitical": lambda: geopolitical_agent(company, geo_indicators, news),
            "insider":      lambda: insider_agent(company, insider_data),
        }

        # ลองโหลด checkpoint ก่อน (ถ้า fail กลางคัน)
        existing_ckpt = load_checkpoints(ticker_input)
        results       = dict(existing_ckpt)
        remaining     = {k:v for k,v in agent_tasks.items() if k not in results}

        agent_status  = {k: ("✅ (cached)" if k in existing_ckpt else "⏳")
                         for k in agent_tasks}
        status_ph     = st.empty()

        def update_status():
            lines = [f"{v} {k.capitalize()}" for k,v in agent_status.items()]
            status_ph.caption("Agents: " + " · ".join(lines))

        update_status()

        if remaining:
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = {pool.submit(fn): name for name,fn in remaining.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        res = future.result()
                        results[name]      = res
                        agent_status[name] = "✅"
                        save_checkpoint(ticker_input, name, res)  # ⑤ Resumable
                    except Exception as e:
                        results[name]      = f"Agent error: {e}"
                        agent_status[name] = "❌"
                    update_status()

        status_ph.empty()

        fin_result     = results.get("financial",    "")
        mac_result     = results.get("macro",        "")
        news_result    = results.get("news",         "")
        tech_result    = results.get("technical",    "")
        geo_result     = results.get("geopolitical", "")
        insider_result = results.get("insider",      "")

        # ===== CONDITIONAL ROUTING ② =====
        triggers  = check_conditional_triggers(
            results, indicators, insider_data, macro, yield_curve
        )
        cond_results = {}
        eco_str    = eco_cal.get("summary","")

        if triggers:
            st.info(f"🔍 พบ condition พิเศษ: {', '.join(triggers)} — รัน Specialized Agents")
            cond_tasks = {}
            if "distressed" in triggers:
                cond_tasks["distressed"] = lambda: distressed_agent(company, fin_result, eco_str)
            if "squeeze" in triggers:
                cond_tasks["squeeze"]    = lambda: short_squeeze_agent(company, insider_data, tech_result)
            if "recession" in triggers:
                cond_tasks["recession"]  = lambda: recession_risk_agent(company, mac_result, yield_curve, sector_rot)
            if "systemic" in triggers:
                cond_tasks["systemic"]   = lambda: systemic_risk_agent(company, geo_result, eco_str)
            if "oversold" in triggers:
                cond_tasks["oversold"]   = lambda: oversold_agent(company, tech_result, valuation_ctx, mgmt_cred)

            with st.spinner(f"Specialized Agents ({len(cond_tasks)} ตัว)..."):
                with ThreadPoolExecutor(max_workers=5) as pool:
                    cf = {pool.submit(fn): name for name,fn in cond_tasks.items()}
                    for future in as_completed(cf):
                        cond_results[cf[future]] = future.result()

        cond_summary = "\n\n".join([
            f"[{k.upper()}]\n{v}" for k,v in cond_results.items()
        ]) if cond_results else ""

        # ===== HUMAN-IN-THE-LOOP ④ =====
        if not st.session_state.get(f"confirmed_{ticker_input}"):
            st.subheader("ผลวิเคราะห์เบื้องต้นจาก Agents")
            ec1, ec2 = st.columns(2)
            with ec1:
                with st.expander("Financial"): st.markdown(fin_result[:400]+"...")
                with st.expander("Macro"):     st.markdown(mac_result[:400]+"...")
                with st.expander("Geo"):       st.markdown(geo_result[:400]+"...")
            with ec2:
                with st.expander("Technical"): st.markdown(tech_result[:400]+"...")
                with st.expander("News"):      st.markdown(news_result[:400]+"...")
                with st.expander("Insider"):   st.markdown(insider_result[:400]+"...")
            if eco_cal.get("fomc_warning"):
                st.warning(eco_cal["fomc_warning"])
            if cond_summary:
                with st.expander("⚠️ Special Analysis"):
                    st.markdown(cond_summary)

            st.divider()
            hc1, hc2 = st.columns(2)
            if hc1.button("✅ ยืนยัน ให้ Orchestrator สรุป", type="primary", use_container_width=True):
                st.session_state[f"confirmed_{ticker_input}"] = True
                st.rerun()
            if hc2.button("🔄 วิเคราะห์ใหม่ทั้งหมด", use_container_width=True):
                clear_checkpoints(ticker_input)
                st.session_state.pop(f"confirmed_{ticker_input}", None)
                st.rerun()
            st.stop()

        # ===== STREAMING ORCHESTRATOR ③⑤ =====
        st.subheader("CIO Full Report")
        orch_prompt = build_orchestrator_prompt(
            company, fin_result, mac_result, geo_result,
            insider_result, news_result, tech_result,
            eco_str, sector_rot, dcf_scenarios_str, cond_summary
        )
        stream_ph = st.empty()
        initial   = stream_orchestrator(orch_prompt, stream_ph)

        # Self-Reflection
        reflection = run_agent(f"""ตรวจสอบการวิเคราะห์นี้:
{initial}
ตรวจหา: ข้อขัดแย้ง, ความมั่นใจเกินจริง, ข้อมูลที่มองข้าม
ถ้าพบปัญหา → ระบุ ถ้าไม่พบ → "สมเหตุสมผล"
ตอบสั้น ภาษาไทย""", 300)

        needs_revision = any(w in reflection for w in [
            "ขัดแย้ง","ปัญหา","ไม่มีหลักฐาน","มองข้าม","ไม่สมเหตุสมผล"
        ])

        if needs_revision:
            st.caption("🔄 Self-Reflection พบปัญหา — กำลังแก้ไข...")
            revision_prompt = f"""แก้ไขการวิเคราะห์:
{initial}
ข้อบกพร่อง: {reflection}
เขียนใหม่ ตอบเป็นภาษาไทย"""
            stream_ph2 = st.empty()
            final = stream_orchestrator(revision_prompt, stream_ph2)
            final = f"{final}\n\n---\n[Self-Reflection: แก้ไขแล้ว ✅]"
        else:
            final = f"{initial}\n\n---\n[Self-Reflection: ผ่าน ✅]"

        # Reset confirmation สำหรับครั้งหน้า
        st.session_state.pop(f"confirmed_{ticker_input}", None)

        st.session_state["history"][ticker_input] = {
            "company":        company,
            "fin_result":     fin_result,
            "mac_result":     mac_result,
            "geo_result":     geo_result,
            "insider_result": insider_result,
            "news_result":    news_result,
            "tech_result":    tech_result,
            "final":          final,
        }
        # บันทึกลง Supabase ถ้าเชื่อมต่ออยู่
        db_save_analysis(ticker_input, company, fin_result, mac_result,
                         geo_result, insider_result, news_result, tech_result, final)

    # เริ่ม chat history ถ้ายังไม่มี
    if ticker_input not in st.session_state["chat_messages"]:
        st.session_state["chat_messages"][ticker_input] = []

    # ===== TABS =====
    tab_dash, tab_fin, tab_mac, tab_geo, tab_insider, tab_news, tab_tech, tab_full, tab_comp, tab_chat = st.tabs([
        "Dashboard", "Financial", "Macro", "Geopolitical", "Insider", "News", "Technical", "CIO Full Report", "Competitors", "Chat"
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

            st.markdown("**Insider Agent**")
            st.markdown(insider_result[:300] + "...")
            with st.expander("ดูรายละเอียด Insider"):
                st.markdown(insider_result)

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

    with tab_insider:
        st.subheader(f"Insider & Market Structure — {ticker_input}")
        if insider_data:
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Short % of Float",  f"{insider_data['short_pct_float']}%",
                        delta="⚠ สูง" if insider_data['short_pct_float'] > 15 else "ปกติ")
            ic2.metric("Days to Cover",     f"{insider_data['short_ratio']} วัน",
                        delta="⚠ ระวัง" if insider_data['short_ratio'] > 5 else "ปกติ")
            ic3.metric("Shares Short",      f"{insider_data['shares_short']:,}")
            st.divider()

        st.markdown(insider_result)

    with tab_news:
        st.subheader(f"News & Sentiment — {ticker_input}")
        st.markdown(news_result)

    with tab_tech:
        st.subheader(f"Technical Analysis — {ticker_input}")
        if options_data:
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Put/Call Ratio", options_data["pc_ratio"],
                        delta=options_data["pc_signal"])
            oc2.metric("Implied Volatility", f"{options_data['iv']}%",
                        delta=options_data["iv_signal"])
            oc3.metric("Max Pain", f"${options_data['max_pain']:.2f}")
            oc4.metric("Expiry", options_data["expiration"])
            st.divider()
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

[Insider Analysis]
{insider_result}

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
                        fin_result, mac_result, geo_result, insider_result, news_result, tech_result, final,
                        st.session_state["chat_messages"][ticker_input][:-1],
                        earnings_date
                    )
                st.markdown(reply)

            # บันทึกคำตอบ
            st.session_state["chat_messages"][ticker_input].append({
                "role": "assistant", "content": reply
            })
            # sync ลง DB
            db_save_chat(ticker_input, st.session_state["chat_messages"][ticker_input])
