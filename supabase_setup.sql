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

-- Portfolio tables (เพิ่มใหม่)
CREATE TABLE portfolios (
    id           BIGSERIAL PRIMARY KEY,
    ticker       TEXT NOT NULL,
    company_name TEXT,
    shares       FLOAT NOT NULL,
    entry_price  FLOAT NOT NULL,
    entry_date   DATE DEFAULT CURRENT_DATE,
    notes        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE portfolio_transactions (
    id         BIGSERIAL PRIMARY KEY,
    ticker     TEXT NOT NULL,
    action     TEXT NOT NULL,
    shares     FLOAT NOT NULL,
    price      FLOAT NOT NULL,
    amount     FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent Checkpoints (เพิ่มใหม่)
CREATE TABLE agent_checkpoints (
    id         BIGSERIAL PRIMARY KEY,
    ticker     TEXT NOT NULL,
    agent      TEXT NOT NULL,
    result     TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

GRANT ALL ON TABLE agent_checkpoints TO anon;
GRANT ALL ON TABLE agent_checkpoints TO authenticated;
GRANT USAGE ON SEQUENCE agent_checkpoints_id_seq TO anon;
GRANT USAGE ON SEQUENCE agent_checkpoints_id_seq TO authenticated;
