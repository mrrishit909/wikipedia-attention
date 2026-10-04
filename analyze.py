"""Steps 2-4: audit the counts, then ask what shapes attention to 12 Wikipedia topics: seasonality, weekdays, spikes and how fast they fade, how predictable next year's peak is, and whether topics displace each other.

    ./venv/bin/python analyze.py
        -> results/audit.csv agents.csv redirects.csv levels.csv weekday.csv seasonality.csv alignment.csv spikes.csv spikes_summary.csv walkforward.csv walkforward_summary.csv displacement.csv displacement_summary.csv summary.csv

Data: counts/daily_views.csv, daily English-Wikipedia page views (agent "user") of 12 articles, 1 January 2016 to 30 September 2026 (3,926 days). Let v be a topic's daily views, y = ln v.
 Weekday effect: y minus the mean of y over the 7 days centred on it (which holds each weekday once), averaged by weekday and centred on zero; swing = exp(max - min) - 1.
 Seasonality (complete years 2016-2025): the weekday effect is removed from y, then each year's mean, giving d. The seasonal profile of a day is the mean of d on the same calendar date in the other nine years (leave one year out); seasonal R2 = 1 - SSE/SST over all days. Amplitude = exp(max - min) of the mean profile, smoothed over 7 days around the year.
 Alignment: for Easter and US Thanksgiving, which move, the same score over the 61 days around the holiday, predicting a day from the same calendar date (as above) or from the same offset from the holiday in the other nine years.
 Spikes: a day with views at least 5 times the median of the previous 28 days starts an event (baseline B = that median, frozen); the peak is the largest day in the next 14; the excess is views - B on the 29 days from the peak. t10 = days until the excess falls to 10% of the peak excess; half-life = ln 2 / lambda from a least-squares line through ln(excess) from the peak to the last day above 10% (needs 4 or more days, kept only if the line falls and R2 is at least 0.8); the next event is searched for from 29 days after the peak.
 Walk-forward (years 2019-2025, each forecast using only earlier years, at least three): the annual peak = the largest day of the calendar year. Date forecasts: last year's peak date; the peak of the mean log-profile (ln v - ln year median) of earlier years, smoothed 7 days; for Easter and Thanksgiving, the holiday date plus the lower median of earlier peak offsets. Error = days apart, around the 365-day year. Height forecasts: last year's peak; the geometric mean of earlier peaks; last year's median x the geometric mean of earlier peak-to-median ratios; the same with last year's median scaled by its change on the year before. Error = |ln(forecast / actual)|.
 Displacement: weekly sums over 560 blocks of 7 days from 1 January 2016, x = ln(week) - ln(the week 52 before) (508 weeks), Pearson correlations of x between topics, with Bartlett's effective n over 12 lags and a Fisher interval; "partial" removes each topic's regression on the mean of the two evergreen topics' x. The correlation of the plain weekly log levels is shown for comparison.
"""
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
TOPICS = ["Easter", "Halloween", "Christmas", "Thanksgiving", "Influenza", "Measles", "Solar_eclipse_of_April_8,_2024", "FIFA_World_Cup", "Eurovision_Song_Contest", "Olympic_Games", "Japan", "Chicago"]
GROUP = dict(zip(TOPICS, ["seasonal"] * 4 + ["disease"] * 2 + ["event"] * 4 + ["evergreen"] * 2))
REDIRECTS = [("Olympics", "Olympic_Games"), ("2024_solar_eclipse", "Solar_eclipse_of_April_8,_2024"), ("Flu", "Influenza")]
RELATED = [("Halloween", "Thanksgiving"), ("Halloween", "Christmas"), ("Thanksgiving", "Christmas"), ("Easter", "Christmas"), ("Influenza", "Measles"),
           ("FIFA_World_Cup", "Olympic_Games"), ("FIFA_World_Cup", "Eurovision_Song_Contest"), ("Olympic_Games", "Eurovision_Song_Contest")]
RELSET = {frozenset(p) for p in RELATED}
SPIKE_X, MIN_VIEWS, BASE_DAYS, PEAK_DAYS, FIT_DAYS, FLOOR, MIN_FIT, MIN_R2 = 5, 3000, 28, 14, 29, 0.1, 4, 0.8
SEASON_YEARS = list(range(2016, 2026)); WF_YEARS = list(range(2019, 2026)); WINDOW = 30; LAGS = 12
CUM = np.cumsum([0] + [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def load():
    d = pd.read_csv(HERE / "counts" / "daily_views.csv", index_col=0, parse_dates=True)
    return d


def pos(ts):
    """position of a date in a leap-year calendar, 0..365"""
    return int(CUM[ts.month - 1] + ts.day - 1)


def easter(y):
    a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31); return date(y, month, day + 1)


def thanksgiving(y):
    d = date(y, 11, 1); d += timedelta((3 - d.weekday()) % 7); return d + timedelta(21)


def smooth7(p):
    return np.array([np.mean([p[(i + j) % len(p)] for j in range(-3, 4)]) for i in range(len(p))])


def main():
    R.mkdir(exist_ok=True)
    df = load(); idx = df.index; N = len(df); years = idx.year.values; dow = idx.weekday.values
    V = {t: df[t + "|user"].values.astype(float) for t in TOPICS}

    # audit ---------------------------------------------------------------
    rows = []
    for t in TOPICS:
        v = V[t]; a = df[t + "|automated"]; first_auto = a[a > 0].index[0]
        rows.append({"topic": t, "days": int(df[t + "|user"].notna().sum()), "missing_days": int(df[t + "|user"].isna().sum()), "zero_days": int((v == 0).sum()), "min": v.min(), "median": float(np.median(v)), "max": v.max(),
                     "max_date": int(idx[int(np.argmax(v))].strftime("%Y%m%d")), "total_views": v.sum(), "first_automated_date": int(first_auto.strftime("%Y%m%d"))})
    pd.DataFrame(rows).to_csv(R / "audit.csv", index=False)
    rows = []
    for t in TOPICS:
        for y in range(2016, 2027):
            m = years == y; rows.append({"topic": t, "year": y, "user": df[t + "|user"].values[m].sum(), "automated": df[t + "|automated"].values[m].sum(), "spider": df[t + "|spider"].values[m].sum()})
    pd.DataFrame(rows).to_csv(R / "agents.csv", index=False)
    rows = []
    for r, tgt in REDIRECTS:
        s = df[r + "|user"]; rows.append({"redirect": r, "target": tgt, "redirect_views": s.sum(), "target_views": V[tgt].sum(), "ratio": s.sum() / V[tgt].sum(), "days_with_no_data": int(s.isna().sum()),
                                           "redirect_peak": s.max(), "target_peak": V[tgt].max()})
    pd.DataFrame(rows).to_csv(R / "redirects.csv", index=False)
    rows = []
    for t in TOPICS:
        for y in range(2016, 2027):
            m = years == y; mm = m & (idx.month <= 9); rows.append({"topic": t, "year": y, "median_views": float(np.median(V[t][m])), "jan_sep_views": V[t][mm].sum()})
    pd.DataFrame(rows).to_csv(R / "levels.csv", index=False)

    # weekday -------------------------------------------------------------
    WE, rows = {}, []
    for t in TOPICS:
        y = np.log(V[t]); ma = np.convolve(y, np.ones(7) / 7, "valid"); r = y[3:N - 3] - ma; dd = dow[3:N - 3]
        e = np.array([r[dd == k].mean() for k in range(7)]); e = e - e.mean(); WE[t] = e
        rows.append({"topic": t, **{f"e{k}": e[k] for k in range(7)}, "swing_pct": 100 * (math.exp(e.max() - e.min()) - 1), "busiest": int(np.argmax(e)), "quietest": int(np.argmin(e))})
    pd.DataFrame(rows).to_csv(R / "weekday.csv", index=False)

    # seasonality ---------------------------------------------------------
    D, A, rows = {}, {}, []
    sm = np.isin(years, SEASON_YEARS)
    for t in TOPICS:
        y = np.log(V[t]) - WE[t][dow]; d = np.full(N, np.nan)
        for yr in SEASON_YEARS:
            m = years == yr; d[m] = y[m] - y[m].mean()
        D[t] = d; M = np.full((len(SEASON_YEARS), 366), np.nan)
        for i in np.where(sm)[0]: M[years[i] - 2016, pos(idx[i])] = d[i]
        A[t] = M; sse = sst = 0.0; dbar = d[sm].mean()
        for i in np.where(sm)[0]:
            k = years[i] - 2016; others = np.delete(M[:, pos(idx[i])], k); sse += (d[i] - np.nanmean(others)) ** 2; sst += (d[i] - dbar) ** 2
        ps = smooth7(np.nanmean(M, axis=0)); pk = int(np.argmax(ps)); month = int(np.searchsorted(CUM[1:], pk, side="right")) + 1
        rows.append({"topic": t, "seasonal_r2": 1 - sse / sst, "amplitude": math.exp(ps.max() - ps.min()), "profile_peak": month * 100 + (pk - CUM[month - 1] + 1)})
    pd.DataFrame(rows).to_csv(R / "seasonality.csv", index=False)

    # alignment -----------------------------------------------------------
    rows = []
    for t, hol in (("Easter", easter), ("Thanksgiving", thanksgiving)):
        H = {yr: hol(yr) for yr in SEASON_YEARS}; d = D[t]; ix = {ts.date(): i for i, ts in enumerate(idx)}; vals, cal, ali, offs = [], [], [], []
        for k, yr in enumerate(SEASON_YEARS):
            for o in range(-WINDOW, WINDOW + 1):
                dt = H[yr] + timedelta(o); i = ix[dt]; vals.append(d[i])
                cal.append(np.nanmean(np.delete(A[t][:, pos(idx[i])], k)))
                ali.append(np.mean([d[ix[H[j] + timedelta(o)]] for j in SEASON_YEARS if j != yr]))
        vals, cal, ali = map(np.array, (vals, cal, ali)); sst = ((vals - vals.mean()) ** 2).sum()
        for yr in SEASON_YEARS:
            seg = [(V[t][ix[H[yr] + timedelta(o)]], o) for o in range(-WINDOW, WINDOW + 1)]; offs.append(max(seg, key=lambda s: (s[0], -s[1]))[1])
        doy = [date(2001, H[yr].month, H[yr].day).toordinal() for yr in SEASON_YEARS]
        rows.append({"topic": t, "days": len(vals), "calendar_r2": 1 - ((vals - cal) ** 2).sum() / sst, "aligned_r2": 1 - ((vals - ali) ** 2).sum() / sst, "holiday_span_days": max(doy) - min(doy), "median_peak_offset": float(np.median(offs))})
    pd.DataFrame(rows).to_csv(R / "alignment.csv", index=False)

    # spikes --------------------------------------------------------------
    rows = []
    for t in TOPICS:
        v = V[t]; i = BASE_DAYS
        while i < N:
            B = float(np.median(v[i - BASE_DAYS:i]))
            if v[i] >= SPIKE_X * B and v[i] >= MIN_VIEWS:
                p = i + int(np.argmax(v[i:min(i + PEAK_DAYS, N)])); ex = v[p:min(p + FIT_DAYS, N)] - B; t10 = next((k for k in range(1, len(ex)) if ex[k] <= FLOOR * ex[0]), None)
                half_k = next((k for k in range(1, len(ex)) if ex[k] <= 0.5 * ex[0]), None); n = t10 if t10 is not None else len(ex); hl = r2 = None
                if n >= MIN_FIT:
                    k = np.arange(n); ly = np.log(ex[:n]); b, a0 = np.polyfit(k, ly, 1)
                    r2 = 1 - ((ly - (a0 + b * k)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
                    if b < 0 and r2 >= MIN_R2: hl = math.log(2) / -b
                rows.append({"topic": t, "start_date": int(idx[i].strftime("%Y%m%d")), "peak_date": int(idx[p].strftime("%Y%m%d")), "peak_views": v[p], "baseline": B, "ratio": v[p] / B, "t10": t10, "half_k": half_k,
                             "fit_n": n if n >= MIN_FIT else None, "halflife": hl, "fit_r2": r2, "censored": int(t10 is None and len(ex) < FIT_DAYS)})
                i = p + FIT_DAYS
            else: i += 1
    sp = pd.DataFrame(rows); sp.to_csv(R / "spikes.csv", index=False)
    rows = []
    for g in [("topic", t) for t in TOPICS] + [("group", g) for g in ("seasonal", "disease", "event", "evergreen", "all")]:
        s = sp[sp["topic"] == g[1]] if g[0] == "topic" else (sp if g[1] == "all" else sp[sp["topic"].map(GROUP) == g[1]])
        rows.append({"level": g[0], "name": g[1], "events": len(s), "with_t10": int(s["t10"].notna().sum()), "median_t10": s["t10"].median() if s["t10"].notna().any() else None,
                     "t10_within_2": int((s["t10"] <= 2).sum()), "fitted": int(s["halflife"].notna().sum()), "median_halflife": s["halflife"].median() if s["halflife"].notna().any() else None})
    pd.DataFrame(rows).to_csv(R / "spikes_summary.csv", index=False)

    # walk-forward --------------------------------------------------------
    rows = []
    for t in TOPICS:
        v = V[t]; info = {}
        for yr in range(2016, 2026):
            m = np.where(years == yr)[0]; pk = m[int(np.argmax(v[m]))]; base = float(np.median(v[m])); prof = np.full(366, np.nan); pr = np.full(366, np.nan)
            for i in m: prof[pos(idx[i])] = math.log(v[i] / base)
            info[yr] = {"peak": v[pk], "doy": idx[pk].dayofyear, "date": idx[pk].date(), "base": base, "prof": prof}
        for yr in WF_YEARS:
            prev = [k for k in range(2016, yr)]; M = np.array([info[k]["prof"] for k in prev]); ps = smooth7(np.nanmean(M, axis=0)); pp = int(np.argmax(ps)); month = int(np.searchsorted(CUM[1:], pp, side="right")) + 1
            dd = pp - CUM[month - 1] + 1; fd = date(yr, month, dd) if not (month == 2 and dd == 29 and yr % 4) else date(yr, 2, 28); prof_doy = fd.timetuple().tm_yday
            ratio = float(np.exp(np.mean([math.log(info[k]["peak"] / info[k]["base"]) for k in prev]))); b1, b2 = info[yr - 1]["base"], info[yr - 2]["base"]
            row = {"topic": t, "year": yr, "actual_doy": info[yr]["doy"], "actual_peak": info[yr]["peak"], "date_last": info[yr - 1]["doy"], "date_profile": prof_doy, "date_rule": None,
                   "h_last": info[yr - 1]["peak"], "h_mean": float(np.exp(np.mean([math.log(info[k]["peak"]) for k in prev]))), "h_ratio": b1 * ratio, "h_trend": b1 * (b1 / b2) * ratio}
            if t in ("Easter", "Thanksgiving"):
                hol = easter if t == "Easter" else thanksgiving; offs = sorted((info[k]["date"] - hol(k)).days for k in prev); row["date_rule"] = (hol(yr) + timedelta(offs[(len(offs) - 1) // 2])).timetuple().tm_yday
            rows.append(row)
    wf = pd.DataFrame(rows); wf.to_csv(R / "walkforward.csv", index=False)
    circ = lambda a, b: np.minimum(abs(a - b), 365 - abs(a - b)); rows = []
    for name in TOPICS + ["annual topics", "other topics"]:
        s = wf[wf["topic"] == name] if name in TOPICS else wf[wf["topic"].isin(["Easter", "Halloween", "Christmas", "Thanksgiving", "Eurovision_Song_Contest"]) == (name == "annual topics")]
        row = {"topic": name, "years": len(s)}
        for c in ("date_last", "date_profile", "date_rule"):
            row["mae_" + c] = float(circ(s[c], s["actual_doy"]).mean()) if s[c].notna().all() else None
        for c in ("h_last", "h_mean", "h_ratio", "h_trend"): row["mae_" + c] = float(abs(np.log(s[c] / s["actual_peak"])).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(R / "walkforward_summary.csv", index=False)

    # displacement --------------------------------------------------------
    nw = N // 7; L = {t: np.log(V[t][:nw * 7].reshape(nw, 7).sum(1)) for t in TOPICS}; X = {t: L[t][52:] - L[t][:-52] for t in TOPICS}; n = nw - 52
    f = (X["Japan"] + X["Chicago"]) / 2; XP = {t: X[t] - (np.cov(X[t], f)[0, 1] / f.var(ddof=1)) * f for t in TOPICS}
    def acf(x, l): xc = x - x.mean(); return (xc[l:] * xc[:-l]).sum() / (xc ** 2).sum()
    rows = []
    for ia, a in enumerate(TOPICS):
        for b in TOPICS[ia + 1:]:
            r = np.corrcoef(X[a], X[b])[0, 1]; ne = min(n, max(10, n / (1 + 2 * sum(acf(X[a], l) * acf(X[b], l) for l in range(1, LAGS + 1))))); z, se = math.atanh(r), 1 / math.sqrt(ne - 3)
            ev = GROUP[a] == "evergreen" or GROUP[b] == "evergreen"
            rows.append({"a": a, "b": b, "related": int(frozenset((a, b)) in RELSET), "evergreen_pair": int(ev), "r_levels": np.corrcoef(L[a], L[b])[0, 1], "r_yoy": r, "n_eff": ne, "ci_lo": math.tanh(z - 1.96 * se), "ci_hi": math.tanh(z + 1.96 * se),
                         "r_partial": None if ev else np.corrcoef(XP[a], XP[b])[0, 1]})
    dp = pd.DataFrame(rows); dp.to_csv(R / "displacement.csv", index=False); rows = []
    ten = dp[dp["evergreen_pair"] == 0]
    for name, s in (("related", ten[ten["related"] == 1]), ("other pairs of the ten topics", ten[ten["related"] == 0]), ("all 66 pairs", dp)):
        rows.append({"group": name, "pairs": len(s), "mean_r_levels": s["r_levels"].mean(), "mean_r_yoy": s["r_yoy"].mean(), "mean_r_partial": s["r_partial"].mean() if s["r_partial"].notna().any() else None,
                     "share_yoy_negative": float((s["r_yoy"] < 0).mean()), "min_r_yoy": s["r_yoy"].min(), "max_r_yoy": s["r_yoy"].max(), "share_ci_excludes_0": float(((s["ci_lo"] > 0) | (s["ci_hi"] < 0)).mean())})
    pd.DataFrame(rows).to_csv(R / "displacement_summary.csv", index=False)

    # summary -------------------------------------------------------------
    pd.DataFrame([{"key": "days", "value": N}, {"key": "weeks", "value": nw}, {"key": "yoy_weeks", "value": n}, {"key": "null_date_error", "value": float(np.mean([min(k, 365 - k) for k in range(365)]))}]).to_csv(R / "summary.csv", index=False)
    print("analyze.py done:", len(sp), "spikes")


if __name__ == "__main__":
    main()
