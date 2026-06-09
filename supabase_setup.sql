-- รัน SQL นี้ใน Supabase Dashboard → SQL Editor

CREATE TABLE analyses (
    id           BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    company_name TEXT,
    price        FLOAT,
    market_cap_b FLOAT,
    sector       TEXT,
    industry     TEXT,
    fin_result   TEXT,
    mac_result   TEXT,
    geo_result   TEXT,
    insider_result TEXT,
    news_result  TEXT,
    tech_result  TEXT,
    final        TEXT,
    analyzed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id         BIGSERIAL PRIMARY KEY,
    ticker     TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE alerts (
    id           BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    target_price FLOAT NOT NULL,
    direction    TEXT NOT NULL,
    label        TEXT,
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
