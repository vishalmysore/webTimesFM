# Builds current.json for the "Current affairs" tab: oil and Bitcoin prices (FRED) and
# MLB team-strength (Elo) histories plus the live playoff series state (MLB Stats API).
# Usage: python build_current.py ../current.json [season]
# Without a season it uses this year, or last year if this year's season hasn't started.
# Any download failure raises, so a scheduled run fails instead of writing partial data.
import sys, csv, io, json, math, datetime as dt, urllib.request

OUT = sys.argv[1]
SEASON = int(sys.argv[2]) if len(sys.argv) > 2 else dt.date.today().year
UA = {"User-Agent": "webTimesFM data builder (github.com/vishalmysore/webTimesFM)"}
get = lambda url: urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()

# ---- prices (FRED) ----
MARKETS = [
    ("DCOILBRENTEU", "brent", "Brent crude oil", "US$ per barrel", 30,
     "Europe's benchmark oil price, daily (weekdays). Pushed up by the Iran war; a pure pattern model can't know whether the conflict escalates or ends."),
    ("DCOILWTICO", "wti", "WTI crude oil", "US$ per barrel", 30,
     "The US benchmark oil price, daily (weekdays). Usually moves with Brent but trades at a discount."),
    ("CBBTCUSD", "bitcoin", "Bitcoin", "US$", 30,
     "Coinbase Bitcoin price, daily including weekends. Markets are watching the Federal Reserve decision on 28 October."),
]
markets = []
for sid, mid, label, unit, h, blurb in MARKETS:
    rows = [(a, float(b)) for a, b in list(csv.reader(io.StringIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"))))[1:] if b not in ("", ".")]
    rows = rows[-1500:]
    assert len(rows) > 200, f"{sid}: only {len(rows)} rows"
    markets.append(dict(id=mid, label=label, group="Current affairs", unit=unit, freq="daily", h=h, ctx=1024, holdout=False,
                        start=rows[0][0], dates=[a for a, _ in rows], values=[round(v, 2) for _, v in rows], blurb=blurb,
                        source=f"FRED series {sid}", url=f"https://fred.stlouisfed.org/series/{sid}"))
    print(f"{mid:8s} {len(rows)} pts, last {rows[-1]}")

# ---- MLB Elo (FiveThirtyEight-style: K=4, home advantage 24, log margin-of-victory weight) ----
K, HFA = 4.0, 24.0
def season_games(year):
    d = json.loads(get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-03-01&endDate={min(dt.date.today(), dt.date(year, 12, 31))}&gameType=R,F,D,L,W"))
    games = sorted([g for x in d["dates"] for g in x["games"]], key=lambda g: g["gameDate"])
    return games, [g for g in games if g["status"]["detailedState"] in ("Final", "Completed Early") and g["teams"]["home"].get("score") is not None]
games, done = season_games(SEASON)
if len(sys.argv) <= 2 and not any(g["gameType"] in "FDLW" for g in done):   # no postseason yet: show last year's
    SEASON -= 1; games, done = season_games(SEASON)
assert len(done) > 500, f"only {len(done)} finished games in {SEASON}"
elo, hist, rec = {}, {}, {}
for g in done:
    A, H = g["teams"]["away"]["team"]["name"], g["teams"]["home"]["team"]["name"]
    sa, sh = g["teams"]["away"]["score"], g["teams"]["home"]["score"]
    for t in (A, H): elo.setdefault(t, 1500.0); hist.setdefault(t, [1500.0]); rec.setdefault(t, [0, 0])
    ph = 1 / (1 + 10 ** (-(elo[H] + HFA - elo[A]) / 400)); hw = sh > sa
    delta = K * math.log(abs(sh - sa) + 1) * ((1.0 if hw else 0.0) - ph)
    elo[H] += delta; elo[A] -= delta; hist[H].append(round(elo[H], 2)); hist[A].append(round(elo[A], 2))
    if g["gameType"] == "R": rec[H][0 if hw else 1] += 1; rec[A][1 if hw else 0] += 1

# series state for the league championship series (L) and World Series (W):
# pairs, wins, and who hosted game 1 (= home-field advantage)
done_ids = {g["gamePk"] for g in done}
def series_state(kind):
    series = {}
    for g in [g for g in games if g["gameType"] == kind]:
        A, H = g["teams"]["away"]["team"]["name"], g["teams"]["home"]["team"]["name"]
        s = series.setdefault(tuple(sorted((A, H))), {"home_field": H, "wins": {A: 0, H: 0}, "pks": set()})
        if g["gamePk"] in done_ids and g["gamePk"] not in s["pks"]:   # count each game once (suspended games repeat)
            s["pks"].add(g["gamePk"]); s["wins"][H if g["teams"]["home"]["score"] > g["teams"]["away"]["score"] else A] += 1
    out = []
    for (a, b), s in series.items():
        hi = s["home_field"]; lo = b if hi == a else a
        out.append({"home_field": hi, "other": lo, "wins": [s["wins"][hi], s["wins"][lo]]})
        print(kind, hi, s["wins"][hi], "-", s["wins"][lo], lo)
    return out
lcs, ws = series_state("L"), series_state("W")
assert len(lcs) == 2, f"expected 2 league championship series, found {len(lcs)}"
teams = sorted({t for s in lcs for t in (s["home_field"], s["other"])})
mlb = dict(season=SEASON, k=K, hfa=HFA, as_of=done[-1]["officialDate"], lcs=lcs, ws=ws[0] if ws else None,
           teams={t: {"elo": round(elo[t], 1), "record": rec[t], "history": hist[t]} for t in teams},
           top=[[t, round(elo[t])] for t in sorted(elo, key=elo.get, reverse=True)[:8]],
           source="MLB Stats API (statsapi.mlb.com), every regular-season and postseason game")
tmp = OUT + ".tmp"
json.dump({"generated": str(dt.date.today()), "markets": markets, "mlb": mlb}, open(tmp, "w"), separators=(",", ":"))
import os; os.replace(tmp, OUT)
print("wrote", OUT, "as of", mlb["as_of"])
