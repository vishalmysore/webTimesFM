# Rebuilds examples.json from public sources.
# Download the raw files into a folder first, then run:  python build_examples.py <folder> ../examples.json
#   covid_us.json  https://disease.sh/v3/covid-19/historical/usa?lastdays=all
#   pov.json       https://api.worldbank.org/v2/country/WLD/indicator/SI.POV.DDAY?format=json&per_page=200
#   <ID>.csv       https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>  for PWHEAMTUSDM, PMAIZMTUSDM, UNRATE
#   co2.csv        https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv
#   sunspots.csv   https://www.sidc.be/SILSO/INFO/snmtotcsv.php
import json, csv, sys, datetime as dt
D = sys.argv[1]; OUT = sys.argv[2]
ex = []
# COVID-19 US daily new cases, 7-day average (JHU via disease.sh)
c = json.load(open(f"{D}/covid_us.json"))["timeline"]["cases"]
dates = [dt.datetime.strptime(k, "%m/%d/%y").date() for k in c]; cum = list(c.values())
new = [max(0, cum[i] - cum[i-1]) for i in range(1, len(cum))]
avg = [round(sum(new[max(0,i-6):i+1]) / len(new[max(0,i-6):i+1])) for i in range(len(new))]
ex.append(dict(id="covid_us", label="COVID-19 daily cases, USA", group="Health", unit="new cases / day (7-day avg)", freq="daily",
  start=str(dates[1]), values=avg, ctx=768, h=60,
  source="Johns Hopkins CSSE via disease.sh", url="https://disease.sh/v3/covid-19/historical/usa?lastdays=all",
  blurb="Waves driven by new variants are hard to predict. Watch how wide the q10–q90 band gets: the model knows it can't see the next wave coming."))
# World poverty headcount ratio, annual (World Bank)
p = json.load(open(f"{D}/pov.json"))[1]
pts = sorted((int(r["date"]), r["value"]) for r in p if r["value"] is not None)
ind = json.load(open(f"{D}/pov.json"))[1][0]["indicator"]["value"]
ex.append(dict(id="poverty_world", label="Extreme poverty rate, world", group="Society", unit="% of population", freq="yearly",
  start=str(pts[0][0]), values=[round(v, 2) for _, v in pts], years=[y for y, _ in pts], ctx=64, h=5,
  source=f"World Bank, {ind} (SI.POV.DDAY)", url="https://data.worldbank.org/indicator/SI.POV.DDAY",
  blurb="A short yearly series (43 points; the source skips 2019). With 5 years held back, the model sees the steady fall up to 2018 and expects it to continue. The real line shows COVID-19 stalling progress in 2020, a shock no model could read from the history."))
def fred(id_):
    rows = list(csv.reader(open(f"{D}/{id_}.csv")))[1:]
    rows = [(a, float(b)) for a, b in rows if b not in ("", ".")]
    return rows[0][0], [round(v, 2) for _, v in rows]
s, v = fred("PWHEAMTUSDM")
ex.append(dict(id="wheat", label="Global wheat price", group="Farming", unit="US$ per metric ton", freq="monthly", start=s, values=v, ctx=320, h=24,
  source="IMF Primary Commodity Prices via FRED (PWHEAMTUSDM)", url="https://fred.stlouisfed.org/series/PWHEAMTUSDM",
  blurb="Crop prices jump on droughts, wars and export bans, such as the 2008 food crisis and the 2022 spike. Mostly a random walk with shocks."))
s, v = fred("PMAIZMTUSDM")
ex.append(dict(id="maize", label="Global maize (corn) price", group="Farming", unit="US$ per metric ton", freq="monthly", start=s, values=v, ctx=320, h=24,
  source="IMF Primary Commodity Prices via FRED (PMAIZMTUSDM)", url="https://fred.stlouisfed.org/series/PMAIZMTUSDM",
  blurb="Compare it with wheat: the two grains often move together because farmers and buyers can switch between them."))
s, v = fred("UNRATE")
ex.append(dict(id="unemployment", label="US unemployment rate", group="Economy", unit="% of labour force", freq="monthly", start=s, values=v, ctx=512, h=24,
  source="US Bureau of Labor Statistics via FRED (UNRATE)", url="https://fred.stlouisfed.org/series/UNRATE",
  blurb="Slow drifts broken by recessions; the April 2020 jump to 14.8% is in the history. No model can predict a recession date from the series alone."))
rows = [r for r in csv.reader(l for l in open(f"{D}/co2.csv") if not l.startswith("#"))][1:]
ex.append(dict(id="co2", label="CO₂ at Mauna Loa", group="Climate", unit="ppm", freq="monthly", start=f"{rows[0][0]}-{int(rows[0][1]):02d}-01",
  values=[float(r[3]) for r in rows], ctx=512, h=60,
  source="NOAA Global Monitoring Laboratory", url="https://gml.noaa.gov/ccgg/trends/data.html",
  blurb="A steady rise plus a yearly 'breathing' cycle as northern-hemisphere plants grow and decay. The easiest series here: expect a tight band."))
rows = [l.split(";") for l in open(f"{D}/sunspots.csv") if l.strip()]
rows = [(int(r[0]), int(r[1]), float(r[3])) for r in rows if float(r[3]) >= 0]
ex.append(dict(id="sunspots", label="Sunspots", group="Science", unit="monthly mean sunspot number", freq="monthly", start=f"{rows[0][0]}-{rows[0][1]:02d}-01",
  values=[v for *_, v in rows], ctx=1024, h=120,
  source="SILSO, Royal Observatory of Belgium", url="https://www.sidc.be/SILSO/datafiles",
  blurb="The ~11-year solar cycle. Give the model a long context (1024+ months) so it sees several cycles."))
for e in ex:
    assert len(e["values"]) > e["h"] + 8, e["id"]
    print(f'{e["id"]:14s} n={len(e["values"]):5d} start={e["start"]} last={e["values"][-1]}')
json.dump({"generated": str(dt.date.today()), "examples": ex}, open(OUT, "w"), separators=(",", ":"))
