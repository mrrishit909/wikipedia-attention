"""Step 5: the project's check. Recompute with different code, in plain Python (json, csv, math, datetime and loops: no pandas, numpy or scipy), and compare with results/:

    ./venv/bin/python check.py

Input: the raw API answers in data/*.json when they are there (run download.py), otherwise counts/daily_views.csv, the committed copy; when both exist every cell of the CSV is compared with the JSON first.
Recomputed: the audit of each series, the agent split by year, the redirects, the yearly levels, the weekday effects, the seasonal R2 and amplitude of each topic, the calendar-against-holiday alignment of Easter and Thanksgiving (the dates by Gauss's
algorithm and by counting Thursdays, not the formulas of analyze.py), every spike with its t10, half-life and fit, the walk-forward forecasts and their errors, and all 66 pairwise correlations of the weekly year-on-year changes with their effective n,
interval and partial correlation. Fits are closed-form least squares. Comparisons are at a relative tolerance of 1e-9 (plus 1e-12 absolute), counts and dates exactly.
"""
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
TOPICS = ["Easter", "Halloween", "Christmas", "Thanksgiving", "Influenza", "Measles", "Solar_eclipse_of_April_8,_2024", "FIFA_World_Cup", "Eurovision_Song_Contest", "Olympic_Games", "Japan", "Chicago"]
GROUP = {t: g for t, g in zip(TOPICS, ["seasonal"] * 4 + ["disease"] * 2 + ["event"] * 4 + ["evergreen"] * 2)}
REDIRECTS = [("Olympics", "Olympic_Games"), ("2024_solar_eclipse", "Solar_eclipse_of_April_8,_2024"), ("Flu", "Influenza")]
RELATED = [("Halloween", "Thanksgiving"), ("Halloween", "Christmas"), ("Thanksgiving", "Christmas"), ("Easter", "Christmas"), ("Influenza", "Measles"),
           ("FIFA_World_Cup", "Olympic_Games"), ("FIFA_World_Cup", "Eurovision_Song_Contest"), ("Olympic_Games", "Eurovision_Song_Contest")]
MDAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
D0 = date(2016, 1, 1)
n_ok = 0


def same(label, got, want, rel=1e-9):
    global n_ok
    n_ok += 1
    if want is None or want == "":
        assert got is None or (isinstance(got, float) and got != got), f"{label}: recount {got} but results is blank"; return
    assert got is not None, f"{label}: recount is blank but results has {want}"
    if isinstance(want, str) and not _num(want): assert got == want, f"{label}: {got} != {want}"; return
    want = float(want)
    if isinstance(got, int): assert got == want, f"{label}: recount {got} != results {want}"; return
    assert abs(got - want) <= rel * abs(want) + 1e-12, f"{label}: recount {got} != results {want}"


def _num(s):
    try: float(s); return True
    except ValueError: return False


def compare(name, rows, keys=None):
    want = list(csv.DictReader(open(R / f"{name}.csv")))
    assert len(want) == len(rows), f"{name}: {len(rows)} rows recounted, {len(want)} in results"
    for i, (w, g) in enumerate(zip(want, rows)):
        for k in (keys or w.keys()): same(f"{name}[{i}].{k}", g[k], w[k])


def mean(v):
    return sum(v) / len(v)


def median(v):
    s = sorted(v); n = len(s); return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def code(d):
    return d.year * 10000 + d.month * 100 + d.day


def pos(d):
    """position of a date in a leap-year calendar"""
    return sum(MDAYS[:d.month - 1]) + d.day - 1


def pos_md(p):
    m = 0
    while p >= MDAYS[m]: p -= MDAYS[m]; m += 1
    return m + 1, p + 1


def smooth7(p):
    n = len(p); return [sum(p[(i + j) % n] for j in range(-3, 4)) / 7 for i in range(n)]


def argmax(v):
    b = 0
    for i, x in enumerate(v):
        if x > v[b]: b = i
    return b


def easter_gauss(y):
    a, b, c, k = y % 19, y % 4, y % 7, y // 100; p = (13 + 8 * k) // 25; q = k // 4; M = (15 - p + k - q) % 30; Nn = (4 + k - q) % 7; d = (19 * a + M) % 30; e = (2 * b + 4 * c + 6 * d + Nn) % 7; days = 22 + d + e
    if d == 29 and e == 6: return date(y, 4, 19)
    if d == 28 and e == 6 and (11 * M + 11) % 30 < 19: return date(y, 4, 18)
    return date(y, 3, days) if days <= 31 else date(y, 4, days - 31)


def thanksgiving_count(y):
    n = 0
    for day in range(1, 31):
        if date(y, 11, day).weekday() == 3:
            n += 1
            if n == 4: return date(y, 11, day)


def load():
    """series[(article, agent)] = list of N counts (None where the API gave no day)"""
    N = (date(2026, 9, 30) - D0).days + 1; days = [D0 + timedelta(i) for i in range(N)]
    rows = list(csv.reader(open(HERE / "counts" / "daily_views.csv"))); head = rows[0]; assert [r[0] for r in rows[1:]] == [d.isoformat() for d in days]
    csvs = {tuple(c.split("|")): [None if r[j] == "" else int(float(r[j])) for r in rows[1:]] for j, c in enumerate(head) if j}
    if not D.exists() or not list(D.glob("*.json")): print("check.py: data/ not found, using counts/daily_views.csv"); return days, csvs
    ser = {}
    for key, col in csvs.items():
        items = json.load(open(D / f"{key[0]}__{key[1]}.json"))["items"]; by = {it["timestamp"][:8]: it["views"] for it in items}; assert len(by) == len(items)
        ser[key] = [by.get(d.strftime("%Y%m%d")) for d in days]; assert ser[key] == col, key
        assert set(by) <= {d.strftime("%Y%m%d") for d in days}
    print(f"check.py: the raw JSON and the committed CSV agree in all {len(ser) * N:,} cells")
    return days, ser


def main():
    days, S = load(); N = len(days); yr = [d.year for d in days]; dw = [d.weekday() for d in days]
    V = {t: [float(x) for x in S[(t, "user")]] for t in TOPICS}
    z = lambda key: [0 if x is None else x for x in S[key]]

    # audit, agents, redirects, levels -----------------------------------
    rows = []
    for t in TOPICS:
        v = V[t]; a = z((t, "automated")); fa = next(i for i in range(N) if a[i] > 0)
        rows.append({"topic": t, "days": N, "missing_days": sum(1 for x in S[(t, "user")] if x is None), "zero_days": sum(1 for x in v if x == 0), "min": min(v), "median": median(v), "max": max(v), "max_date": code(days[v.index(max(v))]), "total_views": sum(v), "first_automated_date": code(days[fa])})
    compare("audit", rows)
    rows = []
    for t in TOPICS:
        us, au, sp = z((t, "user")), z((t, "automated")), z((t, "spider"))
        for y in range(2016, 2027):
            ix = [i for i in range(N) if yr[i] == y]; rows.append({"topic": t, "year": y, "user": sum(us[i] for i in ix), "automated": sum(au[i] for i in ix), "spider": sum(sp[i] for i in ix)})
    compare("agents", rows)
    rows = []
    for r, tgt in REDIRECTS:
        s = z((r, "user")); got = [x for x in S[(r, "user")] if x is not None]; rows.append({"redirect": r, "target": tgt, "redirect_views": sum(got), "target_views": sum(V[tgt]), "ratio": sum(got) / sum(V[tgt]), "days_with_no_data": N - len(got), "redirect_peak": max(got), "target_peak": max(V[tgt])})
    compare("redirects", rows)
    rows = []
    for t in TOPICS:
        for y in range(2016, 2027):
            ix = [i for i in range(N) if yr[i] == y]; rows.append({"topic": t, "year": y, "median_views": median([V[t][i] for i in ix]), "jan_sep_views": sum(V[t][i] for i in ix if days[i].month <= 9)})
    compare("levels", rows)

    # weekday --------------------------------------------------------------
    WE, rows = {}, []
    for t in TOPICS:
        y = [math.log(x) for x in V[t]]; acc = [[] for _ in range(7)]
        for i in range(3, N - 3): acc[dw[i]].append(y[i] - sum(y[i - 3:i + 4]) / 7)
        e = [mean(a) for a in acc]; m = mean(e); e = [x - m for x in e]; WE[t] = e
        rows.append({"topic": t, **{f"e{k}": e[k] for k in range(7)}, "swing_pct": 100 * (math.exp(max(e) - min(e)) - 1), "busiest": argmax(e), "quietest": argmax([-x for x in e])})
    compare("weekday", rows)

    # seasonality ----------------------------------------------------------
    YRS = range(2016, 2026); Dd, M, rows = {}, {}, []
    sidx = [i for i in range(N) if yr[i] <= 2025]
    for t in TOPICS:
        y = [math.log(V[t][i]) - WE[t][dw[i]] for i in range(N)]; d = {}
        for Y in YRS:
            ix = [i for i in range(N) if yr[i] == Y]; mu = mean([y[i] for i in ix])
            for i in ix: d[i] = y[i] - mu
        Dd[t] = d; M[t] = {Y: {} for Y in YRS}
        for i in sidx: M[t][yr[i]][pos(days[i])] = d[i]
        dbar = mean([d[i] for i in sidx]); sse = sst = 0.0
        for i in sidx:
            oth = [M[t][Y][pos(days[i])] for Y in YRS if Y != yr[i] and pos(days[i]) in M[t][Y]]; sse += (d[i] - mean(oth)) ** 2; sst += (d[i] - dbar) ** 2
        prof = []
        for p in range(366): vals = [M[t][Y][p] for Y in YRS if p in M[t][Y]]; prof.append(mean(vals))
        ps = smooth7(prof); pk = argmax(ps); mo, dd = pos_md(pk)
        rows.append({"topic": t, "seasonal_r2": 1 - sse / sst, "amplitude": math.exp(max(ps) - min(ps)), "profile_peak": mo * 100 + dd})
    compare("seasonality", rows)

    # alignment ------------------------------------------------------------
    ix_of = {d: i for i, d in enumerate(days)}; rows = []
    for t, hol in (("Easter", easter_gauss), ("Thanksgiving", thanksgiving_count)):
        H = {Y: hol(Y) for Y in YRS}; d = Dd[t]; vals, cal, ali = [], [], []
        for Y in YRS:
            for o in range(-30, 31):
                dt = H[Y] + timedelta(o); i = ix_of[dt]; vals.append(d[i])
                cal.append(mean([M[t][j][pos(dt)] for j in YRS if j != Y and pos(dt) in M[t][j]])); ali.append(mean([d[ix_of[H[j] + timedelta(o)]] for j in YRS if j != Y]))
        mu = mean(vals); sst = sum((x - mu) ** 2 for x in vals); offs = []
        for Y in YRS:
            best, bo = None, None
            for o in range(-30, 31):
                x = V[t][ix_of[H[Y] + timedelta(o)]]
                if best is None or x > best: best, bo = x, o
            offs.append(bo)
        doy = [date(2001, H[Y].month, H[Y].day).toordinal() for Y in YRS]
        rows.append({"topic": t, "days": len(vals), "calendar_r2": 1 - sum((a - b) ** 2 for a, b in zip(vals, cal)) / sst, "aligned_r2": 1 - sum((a - b) ** 2 for a, b in zip(vals, ali)) / sst, "holiday_span_days": max(doy) - min(doy), "median_peak_offset": median(offs)})
    assert [easter_gauss(Y) for Y in YRS] == [date(*x) for x in [(2016, 3, 27), (2017, 4, 16), (2018, 4, 1), (2019, 4, 21), (2020, 4, 12), (2021, 4, 4), (2022, 4, 17), (2023, 4, 9), (2024, 3, 31), (2025, 4, 20)]]
    compare("alignment", rows)

    # spikes ---------------------------------------------------------------
    sp_rows = []
    for t in TOPICS:
        v = V[t]; i = 28
        while i < N:
            B = median(v[i - 28:i])
            if v[i] >= 5 * B and v[i] >= 3000:
                p = argmax(v[i:min(i + 14, N)]) + i; ex = [x - B for x in v[p:min(p + 29, N)]]; t10 = half_k = None
                for k in range(1, len(ex)):
                    if t10 is None and ex[k] <= 0.1 * ex[0]: t10 = k
                    if half_k is None and ex[k] <= 0.5 * ex[0]: half_k = k
                n = t10 if t10 is not None else len(ex); hl = r2 = None
                if n >= 4:
                    ly = [math.log(x) for x in ex[:n]]; kb = (n - 1) / 2; lb = mean(ly); b = sum((k - kb) * (ly[k] - lb) for k in range(n)) / sum((k - kb) ** 2 for k in range(n)); a0 = lb - b * kb
                    r2 = 1 - sum((ly[k] - a0 - b * k) ** 2 for k in range(n)) / sum((x - lb) ** 2 for x in ly)
                    if b < 0 and r2 >= 0.8: hl = math.log(2) / -b
                sp_rows.append({"topic": t, "start_date": code(days[i]), "peak_date": code(days[p]), "peak_views": v[p], "baseline": B, "ratio": v[p] / B, "t10": t10, "half_k": half_k, "fit_n": n if n >= 4 else None, "halflife": hl, "fit_r2": r2,
                                "censored": int(t10 is None and len(ex) < 29)})
                i = p + 29
            else: i += 1
    compare("spikes", sp_rows)
    rows = []
    for lvl, name in [("topic", t) for t in TOPICS] + [("group", g) for g in ("seasonal", "disease", "event", "evergreen", "all")]:
        s = [r for r in sp_rows if (r["topic"] == name if lvl == "topic" else name == "all" or GROUP[r["topic"]] == name)]; t10s = [r["t10"] for r in s if r["t10"] is not None]; hls = [r["halflife"] for r in s if r["halflife"] is not None]
        rows.append({"level": lvl, "name": name, "events": len(s), "with_t10": len(t10s), "median_t10": median(t10s) if t10s else None, "t10_within_2": sum(1 for x in t10s if x <= 2), "half_next_day": sum(1 for r in s if r["half_k"] == 1), "fitted": len(hls), "median_halflife": median(hls) if hls else None})
    compare("spikes_summary", rows)

    # walk-forward ---------------------------------------------------------
    WF, rows = {}, []
    for t in TOPICS:
        info = {}
        for Y in range(2016, 2026):
            ix = [i for i in range(N) if yr[i] == Y]; pk = ix[argmax([V[t][i] for i in ix])]; base = median([V[t][i] for i in ix])
            info[Y] = {"peak": V[t][pk], "doy": days[pk].timetuple().tm_yday, "date": days[pk], "base": base, "prof": {pos(days[i]): math.log(V[t][i] / base) for i in ix}}
        for Y in range(2019, 2026):
            prev = list(range(2016, Y)); prof = []
            for p in range(366): vals = [info[k]["prof"][p] for k in prev if p in info[k]["prof"]]; prof.append(mean(vals))
            pp = argmax(smooth7(prof)); mo, dd = pos_md(pp); fd = date(Y, 2, 28) if (mo == 2 and dd == 29 and Y % 4) else date(Y, mo, dd)
            ratio = math.exp(mean([math.log(info[k]["peak"] / info[k]["base"]) for k in prev])); b1, b2 = info[Y - 1]["base"], info[Y - 2]["base"]
            row = {"topic": t, "year": Y, "actual_doy": info[Y]["doy"], "actual_peak": info[Y]["peak"], "date_last": (date(Y, 2, 28) if (info[Y - 1]["date"].month, info[Y - 1]["date"].day) == (2, 29) and Y % 4 else date(Y, info[Y - 1]["date"].month, info[Y - 1]["date"].day)).timetuple().tm_yday, "date_profile": fd.timetuple().tm_yday, "date_rule": None, "h_last": info[Y - 1]["peak"],
                   "h_mean": math.exp(mean([math.log(info[k]["peak"]) for k in prev])), "h_ratio": b1 * ratio, "h_trend": b1 * (b1 / b2) * ratio}
            if t in ("Easter", "Thanksgiving"):
                hol = easter_gauss if t == "Easter" else thanksgiving_count; offs = sorted((info[k]["date"] - hol(k)).days for k in prev); row["date_rule"] = (hol(Y) + timedelta(offs[(len(offs) - 1) // 2])).timetuple().tm_yday
            rows.append(row)
    compare("walkforward", rows); WF = rows
    circ = lambda a, b: min(abs(a - b), 365 - abs(a - b)); rows = []; ANN = ("Easter", "Halloween", "Christmas", "Thanksgiving", "Eurovision_Song_Contest")
    for name in TOPICS + ["annual topics", "other topics"]:
        s = [r for r in WF if (r["topic"] == name if name in TOPICS else (r["topic"] in ANN) == (name == "annual topics"))]; row = {"topic": name, "years": len(s)}
        for c in ("date_last", "date_profile", "date_rule"): row["mae_" + c] = mean([circ(r[c], r["actual_doy"]) for r in s]) if all(r[c] is not None for r in s) else None
        for c in ("h_last", "h_mean", "h_ratio", "h_trend"): row["mae_" + c] = mean([abs(math.log(r[c] / r["actual_peak"])) for r in s])
        rows.append(row)
    compare("walkforward_summary", rows)

    # displacement ---------------------------------------------------------
    nw = N // 7; L = {t: [math.log(sum(V[t][7 * k:7 * k + 7])) for k in range(nw)] for t in TOPICS}; X = {t: [L[t][k] - L[t][k - 52] for k in range(52, nw)] for t in TOPICS}; n = nw - 52
    fct = [(a + b) / 2 for a, b in zip(X["Japan"], X["Chicago"])]; fm = mean(fct); fv = sum((x - fm) ** 2 for x in fct)
    XP = {}
    for t in TOPICS:
        xm = mean(X[t]); beta = sum((a - xm) * (b - fm) for a, b in zip(X[t], fct)) / fv; XP[t] = [a - beta * b for a, b in zip(X[t], fct)]

    def corr(a, b):
        ma, mb = mean(a), mean(b); return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))

    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); rk = [0.0] * len(x); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]: j += 1
            for q in range(i, j + 1): rk[order[q]] = (i + j) / 2 + 1
            i = j + 1
        return rk

    def acf(x, l):
        m = mean(x); return sum((x[t] - m) * (x[t - l] - m) for t in range(l, len(x))) / sum((v - m) ** 2 for v in x)

    rows = []
    for ia, a in enumerate(TOPICS):
        for b in TOPICS[ia + 1:]:
            r = corr(X[a], X[b]); ne = min(n, max(10, n / (1 + 2 * sum(acf(X[a], l) * acf(X[b], l) for l in range(1, 13))))); zz, se = math.atanh(r), 1 / math.sqrt(ne - 3); ev = "evergreen" in (GROUP[a], GROUP[b])
            rows.append({"a": a, "b": b, "related": int((a, b) in RELATED or (b, a) in RELATED), "evergreen_pair": int(ev), "r_levels": corr(L[a], L[b]), "r_yoy": r, "r_spearman": corr(ranks(X[a]), ranks(X[b])), "n_eff": ne, "ci_lo": math.tanh(zz - 1.96 * se), "ci_hi": math.tanh(zz + 1.96 * se),
                         "r_partial": None if ev else corr(XP[a], XP[b])})
    compare("displacement", rows); dp = rows; out = []
    ten = [r for r in dp if not r["evergreen_pair"]]
    for name, s in (("related", [r for r in ten if r["related"]]), ("other pairs of the ten topics", [r for r in ten if not r["related"]]), ("all 66 pairs", dp)):
        part = [r["r_partial"] for r in s if r["r_partial"] is not None]
        out.append({"group": name, "pairs": len(s), "mean_r_levels": mean([r["r_levels"] for r in s]), "mean_r_yoy": mean([r["r_yoy"] for r in s]), "mean_r_spearman": mean([r["r_spearman"] for r in s]), "mean_r_partial": mean(part) if part else None, "share_yoy_negative": sum(1 for r in s if r["r_yoy"] < 0) / len(s),
                    "min_r_yoy": min(r["r_yoy"] for r in s), "max_r_yoy": max(r["r_yoy"] for r in s), "share_ci_excludes_0": sum(1 for r in s if r["ci_lo"] > 0 or r["ci_hi"] < 0) / len(s)})
    compare("displacement_summary", out)
    compare("summary", [{"key": "days", "value": N}, {"key": "weeks", "value": nw}, {"key": "yoy_weeks", "value": n}, {"key": "null_date_error", "value": mean([min(k, 365 - k) for k in range(365)])}])
    print(f"check.py: {n_ok} recomputed numbers agree with results/; everything is written out in plain Python from the stored daily counts")


if __name__ == "__main__":
    main()
