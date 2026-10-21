"""
AI vs 我：半年模擬交易實驗
===========================
A 帳戶（AI）：每月最後一個交易日收盤後，從股票池挑「過去 N 個月漲最多、而且買得起」的幾檔，
              隔天開盤平均分配資金換股。實驗第一天開盤先建倉。
B 帳戶（我）：實驗第一天開盤，照設定金額買進固定的 ETF，之後一直持有不動。

每次執行都會從實驗第一天開始，用真實歷史股價重新計算一遍（結果固定、可重現），
然後輸出：
    site/index.html   手機看的報告
    history.csv       每天兩個帳戶的淨值紀錄

設定都在 config.json。這是模擬，不會真的下單。

本機測試：
    pip install -r requirements.txt
    python experiment.py              # 用真實股價
    python experiment.py --demo       # 用模擬股價（不需網路）
"""

import argparse
import html
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TPE = timezone(timedelta(hours=8))

NAMES = {
    "0050": "元大台灣50", "2330": "台積電", "2454": "聯發科", "2308": "台達電", "2317": "鴻海",
    "3711": "日月光投控", "2303": "聯電", "1303": "南亞", "2383": "台光電", "3037": "欣興",
    "2881": "富邦金", "2891": "中信金", "3017": "奇鋐", "2882": "國泰金", "2345": "智邦",
    "2887": "台新新光金", "2382": "廣達", "2327": "國巨", "2885": "元大金", "2884": "玉山金",
    "2360": "致茂", "2357": "華碩", "2886": "兆豐金", "3443": "創意", "6669": "緯穎",
    "2412": "中華電", "2883": "凱基金", "2890": "永豐金", "3653": "健策", "3008": "大立光",
    "3231": "緯創", "2059": "川湖", "2408": "南亞科", "2301": "光寶科", "2344": "華邦電",
    "2892": "第一金", "2880": "華南金", "4958": "臻鼎-KY", "2368": "金像電", "1216": "統一",
    "6446": "藥華藥", "3665": "貿聯-KY", "8046": "南電", "2395": "研華", "2449": "京元電子",
    "5880": "合庫金", "7769": "鴻勁", "2603": "長榮", "4904": "遠傳", "3045": "台灣大",
    "6505": "台塑化",
}


# ================================================================ 資料
def _download(tickers, start):
    import yfinance as yf
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=False, actions=True, progress=False,
                              group_by="column", threads=True)
            if not raw.empty:
                return raw
        except Exception as e:                     # 網路暫時不穩就重試
            print(f"下載失敗（第 {attempt + 1} 次）：{e}", file=sys.stderr)
        time.sleep(10)
    return pd.DataFrame()


def _field(raw, name, tickers):
    if raw.empty or name not in raw.columns.get_level_values(0):
        return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)
    df = raw[name]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df


def load_prices(symbols, start):
    """回傳 dict：Open / Close / Adj / Div 四張表（欄位是股票代號），以及抓不到的代號。"""
    out = {k: [] for k in ("Open", "Close", "Adj", "Div")}
    todo = list(symbols)
    for suffix in (".TW", ".TWO"):                # 上市找不到再試上櫃
        if not todo:
            break
        tickers = [s + suffix for s in todo]
        raw = _download(tickers, start)
        if raw.empty:
            continue
        got = []
        for key, field in (("Open", "Open"), ("Close", "Close"), ("Adj", "Adj Close"), ("Div", "Dividends")):
            df = _field(raw, field, tickers).rename(columns=lambda t: t.split(".")[0])
            out[key].append(df)
        closes = out["Close"][-1]
        got = [s for s in todo if s in closes and closes[s].notna().any()]
        for key in out:
            out[key][-1] = out[key][-1][[s for s in got if s in out[key][-1]]]
        todo = [s for s in todo if s not in got]
    if not out["Close"]:
        raise SystemExit("完全抓不到股價，可能是 Yahoo 暫時無法連線，稍後再試。")
    data = {k: pd.concat(v, axis=1).sort_index() for k, v in out.items()}
    data["Div"] = data["Div"].reindex(columns=data["Close"].columns).fillna(0.0)
    data["Adj"] = data["Adj"].reindex(columns=data["Close"].columns).fillna(data["Close"])
    data["Open"] = data["Open"].reindex(columns=data["Close"].columns)
    data = {k: v[~v.index.duplicated()] for k, v in data.items()}
    return data, todo


def load_demo(symbols, start):
    """模擬股價（含一次配息），只用來測試流程。"""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range(start, date.today())
    close = {}
    for s in symbols:
        r = rng.normal(rng.normal(0.0004, 0.0008), 0.018, len(idx))
        close[s] = rng.uniform(20, 900) * np.exp(np.cumsum(r))
    close = pd.DataFrame(close, index=idx)
    opn = close.shift(1) * (1 + rng.normal(0, 0.004, close.shape))
    opn.iloc[0] = close.iloc[0]
    div = pd.DataFrame(0.0, index=idx, columns=symbols)
    div.iloc[-30, :] = (close.iloc[-31] * 0.01).values          # 一個月前配息 1%
    return {"Open": opn, "Close": close, "Adj": close, "Div": div}, []


# ================================================================ 帳戶
class Account:
    def __init__(self, name, cash, fee_rate, min_fee):
        self.name, self.cash = name, float(cash)
        self.initial = float(cash)
        self.fee_rate, self.min_fee = fee_rate, min_fee
        self.shares, self.cost = {}, {}              # 持股、買進總成本（含手續費）
        self.trades, self.history = [], []
        self.fees_paid = 0.0

    def fee(self, amount):
        return max(math.floor(amount * self.fee_rate), self.min_fee) if amount > 0 else 0

    @staticmethod
    def tax(sym, amount):
        return math.floor(amount * (0.001 if sym.startswith("00") else 0.003))

    def max_buy(self, px, budget):
        n = int(budget // (px * (1 + self.fee_rate)))
        while n > 0 and n * px + self.fee(n * px) > budget:
            n -= 1
        return n

    def buy(self, d, sym, n, px, note=""):
        if n <= 0:
            return
        amt = n * px
        f = self.fee(amt)
        self.cash -= amt + f
        self.fees_paid += f
        self.shares[sym] = self.shares.get(sym, 0) + n
        self.cost[sym] = self.cost.get(sym, 0.0) + amt + f
        self.trades.append({"日期": d, "動作": "買", "代號": sym, "股數": n, "價格": px,
                            "金額": round(amt), "費用": f, "說明": note})

    def sell(self, d, sym, n, px, note=""):
        n = min(n, self.shares.get(sym, 0))
        if n <= 0:
            return
        amt = n * px
        f = self.fee(amt) + self.tax(sym, amt)
        self.cash += amt - f
        self.fees_paid += f
        avg = self.cost[sym] / self.shares[sym]
        self.cost[sym] -= avg * n
        self.shares[sym] -= n
        if self.shares[sym] == 0:
            del self.shares[sym], self.cost[sym]
        self.trades.append({"日期": d, "動作": "賣", "代號": sym, "股數": n, "價格": px,
                            "金額": round(amt), "費用": f, "說明": note})

    def dividend(self, d, sym, per_share):
        n = self.shares.get(sym, 0)
        if n > 0 and per_share > 0:
            amt = round(n * per_share)
            self.cash += amt
            self.trades.append({"日期": d, "動作": "股利", "代號": sym, "股數": n, "價格": per_share,
                                "金額": amt, "費用": 0, "說明": "除息入帳"})

    def value(self, prices):
        return self.cash + sum(n * prices[s] for s, n in self.shares.items())


# ================================================================ AI 選股
def momentum_picks(data, i, pool, lookback, top_n, budget_per_stock):
    """第 i 天收盤後選股：過去 lookback 天漲幅最高、且股價買得起至少 1 股的 top_n 檔。"""
    adj, close = data["Adj"], data["Close"]
    if i - lookback < 0:
        return [], {}
    now, past = adj.iloc[i], adj.iloc[i - lookback]
    mom = (now / past - 1).reindex(pool).dropna()
    mom = mom[close.iloc[i].reindex(mom.index).notna()]
    affordable = mom[close.iloc[i][mom.index] <= budget_per_stock]
    top = affordable.sort_values(ascending=False).head(top_n)
    return list(top.index), {"momentum": mom, "affordable": affordable}


def rebalance(acct, d, picks, opens, top_n, note):
    value = acct.cash + sum(n * opens[s] for s, n in acct.shares.items())
    per = value * 0.99 / top_n                       # 留 1% 現金付手續費
    targets = {s: (acct.max_buy(opens[s], per) if s in picks else 0) for s in set(picks) | set(acct.shares)}
    for s, t in targets.items():                     # 先賣再買
        if acct.shares.get(s, 0) > t:
            acct.sell(d, s, acct.shares[s] - t, opens[s], note)
    for s in picks:
        need = targets[s] - acct.shares.get(s, 0)
        if need > 0:
            n = acct.max_buy(opens[s], min(need * opens[s] * 1.01 + acct.min_fee, acct.cash))
            acct.buy(d, s, min(n, need), opens[s], note)


def preview_orders(acct, picks, closes, top_n):
    """用今天收盤價估算明天開盤要做的交易（實際會用明天開盤價）。"""
    value = acct.value(closes)
    per = value * 0.99 / top_n
    orders = []
    for s in sorted(set(acct.shares) - set(picks)):
        orders.append(("賣", s, acct.shares[s]))
    for s in picks:
        t = acct.max_buy(closes[s], per)
        diff = t - acct.shares.get(s, 0)
        if diff > 0:
            orders.append(("買", s, diff))
        elif diff < 0:
            orders.append(("賣", s, -diff))
    return orders


# ================================================================ 主流程
def run(cfg, data):
    start = pd.Timestamp(cfg["start_date"])
    fee_rate = 0.001425 * cfg["fees"]["discount"]
    min_fee = cfg["fees"]["min_fee"]
    ai_cfg, me_cfg = cfg["ai"], cfg["me"]
    top_n, lookback = ai_cfg["top_n"], ai_cfg["lookback_months"] * 21
    pool = [s for s in ai_cfg["pool"] if s in data["Close"]]

    ai = Account("AI", ai_cfg["capital"], fee_rate, min_fee)
    me = Account("我", sum(me_cfg["buy"].values()), fee_rate, min_fee)

    idx = data["Close"].index
    close_f = data["Close"].ffill()
    started = idx[idx >= start]
    result = {"ai": ai, "me": me, "pool": pool, "started": len(started) > 0, "pending": None,
              "last_date": idx[-1].date() if len(idx) else None, "picks_log": []}

    if not len(started):
        # 實驗還沒開始：預告第一天開盤要做的事
        i = len(idx) - 1
        picks, _ = momentum_picks(data, i, pool, lookback, top_n, ai.initial * 0.99 / top_n)
        result["pending"] = {"when": "實驗第一天開盤", "picks": picks,
                             "orders": preview_orders(ai, picks, close_f.iloc[i], top_n),
                             "me_orders": [("買", s, me.max_buy(close_f.iloc[i][s], amt))
                                           for s, amt in me_cfg["buy"].items() if s in close_f]}
        return result

    first = idx.get_loc(started[0])
    periods = idx.to_period("M")
    pending = None
    if first > 0:                                         # 實驗前一天收盤選出第一批
        picks, _ = momentum_picks(data, first - 1, pool, lookback, top_n, ai.initial * 0.99 / top_n)
        pending = picks
        result["picks_log"].append({"選股日": idx[first - 1].date(), "選出": picks, "說明": "建倉"})

    for i in range(first, len(idx)):
        d = idx[i].date()
        opens = data["Open"].iloc[i].fillna(close_f.iloc[i - 1] if i > 0 else close_f.iloc[i])
        closes = close_f.iloc[i]

        for s, per_share in data["Div"].iloc[i].items():  # 除息日入帳股利
            if per_share > 0:
                ai.dividend(d, s, per_share)
                me.dividend(d, s, per_share)

        if i == first:                                    # B 帳戶建倉
            for s, amt in me_cfg["buy"].items():
                if s in opens and pd.notna(opens[s]):
                    budget = min(amt, me.cash)
                    me.buy(d, s, me.max_buy(opens[s], budget), opens[s], "建倉")
        if pending is not None:
            rebalance(ai, d, pending, opens, top_n, "建倉" if i == first else "每月換股")
            pending = None

        ai.history.append((d, ai.value(closes)))
        me.history.append((d, me.value(closes)))

        is_month_end = i + 1 < len(idx) and periods[i + 1] != periods[i]
        if is_month_end:
            picks, _ = momentum_picks(data, i, pool, lookback, top_n, ai.value(closes) * 0.99 / top_n)
            pending = picks
            result["picks_log"].append({"選股日": d, "選出": picks, "說明": "月底選股"})

    # 最後一天如果剛好是月底（或資料停在月底），預告明天的交易
    i = len(idx) - 1
    today_tpe = datetime.now(TPE).date()
    month_end_today = (idx[-1] + pd.offsets.BMonthEnd(0)).date() == idx[-1].date()
    if month_end_today or (today_tpe.month != idx[-1].month and pending is None):
        picks, _ = momentum_picks(data, i, pool, lookback, top_n, ai.value(close_f.iloc[i]) * 0.99 / top_n)
        result["pending"] = {"when": "下一個交易日開盤", "picks": picks,
                             "orders": preview_orders(ai, picks, close_f.iloc[i], top_n), "me_orders": []}
    result["closes"] = close_f.iloc[-1]
    return result


# ================================================================ 報告
def fmt_money(v):
    return f"{v:,.0f}"


def pct(v):
    return f"{v:+.2%}"


def svg_chart(ai_hist, me_hist):
    if len(ai_hist) < 2:
        return '<p class="muted">累積兩天以上的資料後，這裡會出現走勢圖。</p>'
    dates = [d for d, _ in ai_hist]
    a = [v for _, v in ai_hist]
    b = [v for _, v in me_hist]
    lo, hi = min(a + b), max(a + b)
    pad = (hi - lo) * 0.1 or hi * 0.02
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 640, 260, 56, 12, 12, 28
    x = lambda k: L + (W - L - R) * k / (len(dates) - 1)
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    line = lambda vals: " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in enumerate(vals))
    ticks = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
        f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>'
        for v in np.linspace(lo, hi, 5)[1:-1])
    xl = "".join(f'<text x="{x(k):.1f}" y="{H - 8}" text-anchor="{anc}">{dates[k].strftime("%m/%d")}</text>'
                 for k, anc in ((0, "start"), (len(dates) // 2, "middle"), (len(dates) - 1, "end")))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="兩個帳戶淨值走勢">{ticks}{xl}'
            f'<polyline points="{line(b)}" class="ln me"/><polyline points="{line(a)}" class="ln ai"/></svg>')


def holdings_table(acct, closes):
    if not acct.shares:
        return '<p class="muted">目前沒有持股（全部是現金）。</p>'
    rows = []
    for s, n in sorted(acct.shares.items(), key=lambda kv: -kv[1] * closes[kv[0]]):
        mv = n * closes[s]
        pl = mv - acct.cost[s]
        rows.append(f"<tr><td>{s}<small>{html.escape(NAMES.get(s, ''))}</small></td><td>{n}</td>"
                    f"<td>{closes[s]:,.2f}</td><td>{fmt_money(mv)}</td>"
                    f'<td class="{"up" if pl >= 0 else "down"}">{pl:+,.0f}</td></tr>')
    return ('<table><thead><tr><th>股票</th><th>股數</th><th>收盤</th><th>市值</th><th>損益</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def trades_table(acct, limit=30):
    if not acct.trades:
        return '<p class="muted">還沒有交易。</p>'
    rows = "".join(
        f'<tr><td>{t["日期"]:%m/%d}</td><td class="{ {"買": "buy", "賣": "sell"}.get(t["動作"], "div") }">{t["動作"]}</td>'
        f'<td>{t["代號"]}<small>{html.escape(NAMES.get(t["代號"], ""))}</small></td><td>{t["股數"]}</td>'
        f'<td>{t["價格"]:,.2f}</td><td>{fmt_money(t["金額"])}</td><td>{t["費用"]}</td></tr>'
        for t in reversed(acct.trades[-limit:]))
    return ('<table><thead><tr><th>日期</th><th></th><th>股票</th><th>股數</th><th>價格</th><th>金額</th>'
            f'<th>費用</th></tr></thead><tbody>{rows}</tbody></table>')


def orders_list(orders):
    if not orders:
        return "<li>不需要交易</li>"
    return "".join(f'<li><b class="{"buy" if a == "買" else "sell"}">{a}</b> {s} '
                   f'{html.escape(NAMES.get(s, ""))}　'
                   f'{f"{n} 股" if n > 0 else "<span class=muted>金額不夠買 1 股，先放現金</span>"}</li>'
                   for a, s, n in orders)


def render(cfg, res, failed):
    ai, me = res["ai"], res["me"]
    now = datetime.now(TPE).strftime("%Y-%m-%d %H:%M")
    title = "AI vs 我　模擬交易實驗"
    css = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1b1f24;--muted:#6b7280;--line:#e5e7eb;--ai:#2563eb;--me:#d97706;
--up:#d1242f;--down:#1a7f37;--buy:#d1242f;--sell:#1a7f37}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--text:#e6e8eb;--muted:#9aa3ae;
--line:#2a2f37;--ai:#60a5fa;--me:#fbbf24;--up:#ff6b6b;--down:#4ade80;--buy:#ff6b6b;--sell:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 -apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0}h2{font-size:16px;margin:0 0 10px}
.muted{color:var(--muted);font-size:13px}small{display:block;color:var(--muted);font-size:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}
.vs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.acct{border-radius:12px;padding:12px;background:var(--bg)}
.acct .who{font-size:13px;font-weight:600}.ai .who{color:var(--ai)}.me .who{color:var(--me)}
.acct .val{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}.buy{color:var(--buy)}.sell{color:var(--sell)}.div{color:var(--ai)}
.lead{font-size:15px;font-weight:600;margin-top:10px;text-align:center}
svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
.grid{stroke:var(--line)}.ln{fill:none;stroke-width:2.2;stroke-linejoin:round}.ln.ai{stroke:var(--ai)}.ln.me{stroke:var(--me)}
.legend{display:flex;gap:16px;font-size:13px;margin-top:6px}.legend i{display:inline-block;width:14px;height:3px;
vertical-align:middle;margin-right:6px;border-radius:2px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:500;text-align:right;padding:6px 4px;border-bottom:1px solid var(--line)}
td{text-align:right;padding:7px 4px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto}ul{margin:6px 0;padding-left:20px}
details summary{cursor:pointer;font-weight:600}
.note{font-size:12px;color:var(--muted);line-height:1.7}
"""
    head = (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<meta name="theme-color" content="#2563eb"><style>{css}</style></head><body><main>'
            f'<h1>🤖 AI vs 🙋 我</h1><div class="muted">模擬交易實驗・開始日 {cfg["start_date"]}・'
            f'資料到 {res["last_date"]}・更新 {now}</div>')

    warn = ""
    if failed:
        warn = f'<div class="card note">⚠ 抓不到這些代號的股價，已略過：{"、".join(failed)}</div>'

    if not res["started"]:
        p = res["pending"]
        body = (f'<div class="card"><h2>⏳ 實驗還沒開始</h2><p>實驗會在 <b>{cfg["start_date"]}</b> 開盤開始。'
                f'依照最新收盤價，第一天預計：</p>'
                f'<h2 style="color:var(--ai)">AI 帳戶（{fmt_money(ai.initial)} 元）</h2><ul>{orders_list(p["orders"])}</ul>'
                f'<h2 style="color:var(--me)">我的帳戶（{fmt_money(me.initial)} 元）</h2><ul>{orders_list(p["me_orders"])}</ul>'
                f'<p class="note">股數用今天收盤價估算，實際以開盤價計算。</p></div>')
        return head + warn + body + footer(cfg) + "</main></body></html>"

    closes = res["closes"]
    va, vb = ai.history[-1][1], me.history[-1][1]
    ra, rb = va / ai.initial - 1, vb / me.initial - 1
    days = len(ai.history)
    diff = ra - rb
    lead = ("目前打平" if abs(diff) < 0.0005 else
            f'目前 <span style="color:var(--ai)">AI</span> 領先 {abs(diff):.2%}' if diff > 0 else
            f'目前 <span style="color:var(--me)">我</span> 領先 {abs(diff):.2%}')

    pend = ""
    if res["pending"]:
        p = res["pending"]
        pend = (f'<div class="card"><h2>📋 {p["when"]} AI 預計交易</h2><ul>{orders_list(p["orders"])}</ul>'
                f'<p class="note">依今天收盤價估算；實際用開盤價成交。要跟單的話，可以用零股照這張清單下單。</p></div>')

    picks_rows = "".join(f'<tr><td>{p["選股日"]:%m/%d}</td><td>{p["說明"]}</td><td>'
                         f'{"、".join(f"{s} {NAMES.get(s, chr(32))}".strip() for s in p["選出"]) or "（沒有買得起的）"}'
                         f'</td></tr>' for p in reversed(res["picks_log"]))

    body = f"""
<div class="card">
  <div class="vs">
    <div class="acct ai"><div class="who">🤖 AI 帳戶</div><div class="val">{fmt_money(va)}</div>
      <div class="{'up' if ra >= 0 else 'down'}">{pct(ra)}</div><div class="muted">本金 {fmt_money(ai.initial)}</div></div>
    <div class="acct me"><div class="who">🙋 我的帳戶</div><div class="val">{fmt_money(vb)}</div>
      <div class="{'up' if rb >= 0 else 'down'}">{pct(rb)}</div><div class="muted">本金 {fmt_money(me.initial)}</div></div>
  </div>
  <div class="lead">{lead}　<span class="muted">（第 {days} 個交易日）</span></div>
</div>
<div class="card"><h2>淨值走勢</h2>{svg_chart(ai.history, me.history)}
  <div class="legend"><span><i style="background:var(--ai)"></i>AI</span><span><i style="background:var(--me)"></i>我</span></div></div>
{pend}
<div class="card"><h2 style="color:var(--ai)">🤖 AI 持股</h2><div class="scroll">{holdings_table(ai, closes)}</div>
  <p class="muted">現金 {fmt_money(ai.cash)} 元・累計手續費和稅 {fmt_money(ai.fees_paid)} 元</p></div>
<div class="card"><h2 style="color:var(--me)">🙋 我的持股</h2><div class="scroll">{holdings_table(me, closes)}</div>
  <p class="muted">現金 {fmt_money(me.cash)} 元・累計手續費和稅 {fmt_money(me.fees_paid)} 元</p></div>
<div class="card"><details><summary>AI 每次選股</summary><div class="scroll"><table><thead><tr><th>日期</th><th></th>
  <th>選出</th></tr></thead><tbody>{picks_rows}</tbody></table></div></details></div>
<div class="card"><details><summary>AI 交易紀錄</summary><div class="scroll">{trades_table(ai)}</div></details></div>
<div class="card"><details><summary>我的交易紀錄</summary><div class="scroll">{trades_table(me)}</div></details></div>
"""
    return head + warn + body + footer(cfg) + "</main></body></html>"


def footer(cfg):
    ai, f = cfg["ai"], cfg["fees"]
    me = "、".join(f"{s} {v:,} 元" for s, v in cfg["me"]["buy"].items())
    return (f'<div class="card note"><b>實驗規則</b><br>'
            f'AI：每月最後一個交易日收盤後，從 {len(ai["pool"])} 檔股票中，挑過去 {ai["lookback_months"]} 個月漲最多、'
            f'而且買得起至少 1 股的 {ai["top_n"]} 檔，隔天開盤平均分配資金換股（零股）。<br>'
            f'我：第一天開盤買進 {me}，之後不動。<br>'
            f'成本：手續費 0.1425% × {f["discount"]} 折、每筆最低 {f["min_fee"]} 元；賣出證交稅股票 0.3%、ETF 0.1%；'
            f'股利在除息日入帳（未扣二代健保）。<br>'
            f'成交價用開盤價，實際零股成交價可能不同。這是模擬，不是投資建議。</div>')


# ================================================================ 進入點
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用模擬股價測試")
    ap.add_argument("--out", default=str(ROOT / "site" / "index.html"))
    ap.add_argument("--start", help="測試用：暫時改用別的開始日")
    a = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if a.start:
        cfg["start_date"] = a.start
    symbols = list(dict.fromkeys(cfg["ai"]["pool"] + list(cfg["me"]["buy"])))
    fetch_from = (pd.Timestamp(cfg["start_date"]) - pd.Timedelta(days=cfg["ai"]["lookback_months"] * 31 + 60)).date()
    if fetch_from > date.today() - timedelta(days=cfg["ai"]["lookback_months"] * 31 + 60):
        fetch_from = date.today() - timedelta(days=cfg["ai"]["lookback_months"] * 31 + 60)

    data, failed = (load_demo if a.demo else load_prices)(symbols, str(fetch_from))
    res = run(cfg, data)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cfg, res, failed), encoding="utf-8")

    if res["started"]:
        hist = pd.DataFrame({"日期": [d for d, _ in res["ai"].history],
                             "AI淨值": [round(v) for _, v in res["ai"].history],
                             "我的淨值": [round(v) for _, v in res["me"].history]})
        hist.to_csv(ROOT / "history.csv", index=False, encoding="utf-8-sig")
        print(f"AI {hist['AI淨值'].iloc[-1]:,} ／ 我 {hist['我的淨值'].iloc[-1]:,}（{len(hist)} 個交易日）")
    else:
        print("實驗尚未開始，已輸出預告頁。")
    print(f"報告已輸出：{out}")


if __name__ == "__main__":
    main()
