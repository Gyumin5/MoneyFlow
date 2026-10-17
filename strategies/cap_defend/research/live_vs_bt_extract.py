import sqlite3, json
c = sqlite3.connect("/home/ubuntu/assets.db")
out = []
q = "select snapshot_date, accounts_json, fx_rate from snapshots where snapshot_date>=? order by snapshot_date"
for d, aj, fx in c.execute(q, ("2026-04-01",)):
    if not aj:
        continue
    a = json.loads(aj)
    s = a.get("stock_kis") or {}
    u = a.get("coin_upbit") or {}
    b = a.get("coin_binance") or {}
    row = dict(date=d, fx=fx)
    row["st"] = dict(tot_krw=s.get("total_krw"), tot_usd=s.get("total_usd"),
                     cash_krw=s.get("cash_krw"), cash_usd=s.get("cash_usd"),
                     eval_usd=s.get("stock_eval_usd"), fx=s.get("exchange_rate"),
                     h={x["ticker"]: [x.get("qty"), x.get("eval_amt"), x.get("value_krw")]
                        for x in (s.get("holdings") or [])},
                     upd=s.get("updated"))
    row["up"] = dict(tot_krw=u.get("total_krw"), cash_krw=u.get("krw_balance"),
                     h={x["ticker"]: [x.get("qty"), x.get("value"), x.get("price")]
                        for x in (u.get("holdings") or [])},
                     upd=u.get("updated"))
    row["fu"] = dict(tot_usd=b.get("total_usdt"), fut_tot_usd=b.get("futures_total_usdt"),
                     cash_usd=b.get("futures_cash_usdt"), fx=b.get("exchange_rate"),
                     unreal=b.get("unrealized_usdt"),
                     h={x["ticker"]: [x.get("qty"), x.get("notional_usdt"), x.get("value_usdt"),
                                      x.get("leverage"), x.get("entry_price"), x.get("price")]
                        for x in (b.get("holdings") or [])},
                     tw=b.get("target_weights"), upd=b.get("updated"))
    out.append(row)
print(json.dumps(out, ensure_ascii=False))
