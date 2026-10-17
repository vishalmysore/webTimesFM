# Adds a daily news-tone signal to current.json for the "Current affairs" tab.
#
# For each topic (oil, Bitcoin, each MLB team still in the playoffs) it reads up to 100
# Google News headlines per day for the last 14 days, scores each headline with VADER
# (a small, rule-based sentiment model; compound score -1..+1) and averages them into a
# daily tone on a -10..+10 scale. Daily tones are kept in news_history.json so the record
# grows over time.
#
# Tone says how positive or negative the coverage sounds. It does NOT say which way a
# price will move ("war escalates" is negative news but pushes oil up), so the page uses
# it only as a warning light: an alert fires when the latest day's tone is unusually far
# from the topic's own recent normal.
#
# News is optional: if Google News can't be reached, this script leaves current.json
# without a news block and exits 0, so the prices and baseball update still go out.
#
# Usage: python build_news.py ../current.json ../news_history.json
import sys, json, re, html, time, statistics, datetime as dt, email.utils
import urllib.request, urllib.parse, xml.etree.ElementTree as ET

CUR, HIST = sys.argv[1], sys.argv[2]
DAYS, THROTTLE = 14, 1.2
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    sys.exit("needs VADER: python -m pip install vaderSentiment")
vader = SentimentIntensityAnalyzer()

def headlines(query, day):
    q = f"{query} after:{day} before:{day + dt.timedelta(days=1)}"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    xml = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
    out = []
    for it in ET.fromstring(xml).iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title: continue
        clean = html.unescape(re.sub(r"\s+-\s+[^-]+$", "", title))          # drop " - Publisher"
        src = it.find("source")
        pub = (src.text or "").strip() if src is not None else ""
        try: when = email.utils.parsedate_to_datetime(it.findtext("pubDate")).isoformat()
        except Exception: when = ""
        out.append({"title": clean, "source": pub, "url": (it.findtext("link") or "").strip(), "date": when,
                    "tone": round(10 * vader.polarity_scores(clean)["compound"], 2)})
    return out

cur = json.load(open(CUR))
topics = {"oil": ('"crude oil" OR "oil prices" OR "Brent crude" OR "WTI crude" OR OPEC', ["brent", "wti"]),
          "bitcoin": ('bitcoin OR BTC price', ["bitcoin"])}
for team in cur["mlb"]["teams"]:
    topics[team] = (f'"{team}"', [])
try: hist = json.load(open(HIST))
except (FileNotFoundError, json.JSONDecodeError): hist = {}

today = dt.datetime.now(dt.timezone.utc).date()
days = [today - dt.timedelta(days=k) for k in range(DAYS, 0, -1)]       # last 14 full days, oldest first
news, failures = {}, 0
for topic, (query, market_ids) in topics.items():
    h = hist.setdefault(topic, {}); latest = []
    for day in days:
        try: items = headlines(query, day)
        except Exception as e:
            failures += 1; print(f"  {topic} {day}: {e}", file=sys.stderr); time.sleep(THROTTLE); continue
        time.sleep(THROTTLE)
        if items:
            h[str(day)] = {"tone": round(statistics.fmean(i["tone"] for i in items), 2), "n": len(items)}
            if day == days[-1]: latest = items
    series = sorted(h.items())[-30:]                                       # up to 30 days of history for the baseline
    if len(series) < 5: continue
    (last_day, last), base = series[-1], [v["tone"] for _, v in series[:-1]]
    mu, sd = statistics.fmean(base), statistics.pstdev(base) or 0.5
    z, shift = (last["tone"] - mu) / sd, last["tone"] - mu
    alert = abs(z) >= 2 and abs(shift) >= 1.0
    news[topic] = {"query": query, "markets": market_ids, "day": last_day, "tone": last["tone"], "n": last["n"],
                   "baseline": round(mu, 2), "spread": round(sd, 2), "z": round(z, 2), "alert": alert,
                   "reason": (f"news tone {'fell' if shift < 0 else 'rose'} to {last['tone']:+.1f} on {last_day}, "
                              f"{abs(z):.1f}x its usual day-to-day swing (normal {mu:+.1f})") if alert else "",
                   "daily": [{"date": d, "tone": v["tone"], "n": v["n"]} for d, v in series],
                   "headlines": sorted(latest, key=lambda i: abs(i["tone"]), reverse=True)[:5]}
    print(f"{topic:22s} {last_day} tone {last['tone']:+.2f} (normal {mu:+.2f} ± {sd:.2f}) z {z:+.1f} {'ALERT' if alert else ''}")

if not news:
    print(f"no news collected ({failures} failed requests); leaving current.json unchanged"); sys.exit(0)
cur["news"] = {"generated": str(today), "method": "Google News RSS headlines, up to 100 per day, scored with VADER (compound x10 = -10..+10)",
               "topics": news}
json.dump(cur, open(CUR, "w"), separators=(",", ":"))
json.dump(hist, open(HIST, "w"), separators=(",", ":"), sort_keys=True)
print(f"wrote news for {len(news)} topics ({failures} failed requests)")
