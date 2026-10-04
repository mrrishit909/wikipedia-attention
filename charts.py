"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_seasonality.png, 02_alignment.png, 03_spikes.png, 04_walkforward.png, 05_displacement.png"""
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyze import CUM, GROUP, TOPICS, easter, load, pos, smooth7, thanksgiving

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
SHORT = {"Solar_eclipse_of_April_8,_2024": "2024 eclipse", "FIFA_World_Cup": "World Cup", "Eurovision_Song_Contest": "Eurovision", "Olympic_Games": "Olympics"}
nm = lambda t: SHORT.get(t, t)
GCOL = {"seasonal": BLUE, "disease": ORANGE, "event": GREEN, "evergreen": GRAY}
SE = pd.read_csv(R / "seasonality.csv").set_index("topic"); WD = pd.read_csv(R / "weekday.csv").set_index("topic"); AL = pd.read_csv(R / "alignment.csv").set_index("topic"); SP = pd.read_csv(R / "spikes.csv")
WF = pd.read_csv(R / "walkforward_summary.csv").set_index("topic"); DP = pd.read_csv(R / "displacement.csv"); AG = pd.read_csv(R / "agents.csv"); SS = pd.read_csv(R / "summary.csv").set_index("key")["value"]
DF = load(); IDX = DF.index; V = {t: DF[t + "|user"].values.astype(float) for t in TOPICS}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def profile(t):
    y = np.log(V[t]); yrs = IDX.year.values; M = np.full((10, 366), np.nan)
    for i in np.where(yrs <= 2025)[0]:
        m = yrs == yrs[i]; M[yrs[i] - 2016, pos(IDX[i])] = y[i] - y[m].mean()
    return np.exp(smooth7(np.nanmean(M, axis=0)))


def year_profiles(ax, fs=9):
    cols = {"Halloween": BLUE, "Thanksgiving": ORANGE, "Christmas": GREEN, "Easter": "#c9b458", "Eurovision_Song_Contest": "#b06bd0", "Influenza": GRAY}
    for t, c in cols.items():
        p = profile(t); ax.plot(np.arange(366), p, color=c, linewidth=2.2 if t != "Influenza" else 1.6, label=nm(t))
    ax.set_yscale("log"); ax.set_ylim(0.1, 120); ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1x", "1x", "10x", "100x"]); ax.axhline(1, color=DIM, linewidth=0.9, linestyle=":")
    ax.set_xticks(CUM[:12] + 14); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_xlim(0, 365); ax.set_ylabel("views against the topic's year average (log scale)"); ax.legend(frameon=False, labelcolor=INK, fontsize=fs, loc="upper left", ncol=2)
    ax.set_title("A year as Wikipedia sees it: the typical day, 2016 to 2025", fontsize=12)


def t10_hist(ax):
    t = SP["t10"]; lab = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-28", "not within\n29 days"]
    cnt = [int((t == 1).sum()), int((t == 2).sum()), int((t == 3).sum()), int((t == 4).sum()), int((t == 5).sum()), int(t.between(6, 10).sum()), int(t.between(11, 20).sum()), int(t.between(21, 28).sum()), int(t.isna().sum())]
    assert sum(cnt) == len(t); b = ax.bar(range(len(cnt)), cnt, color=[BLUE if k < 5 else GRAY for k in range(9)], width=0.7)
    for r, c in zip(b, cnt): ax.text(r.get_x() + r.get_width() / 2, c + 0.8, str(c), ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(range(len(cnt))); ax.set_xticklabels(lab, fontsize=9); ax.set_xlabel("days after the peak until the excess is 10% of its peak value"); ax.set_ylabel("spikes"); ax.set_title(f"{len(t)} spikes: {int((t <= 3).sum())} are 90% gone within 3 days", fontsize=12)
    ax.set_ylim(0, max(cnt) * 1.12)


def seasonality(ax):
    s = SE["seasonal_r2"].sort_values(ascending=False); ax.bar(range(len(s)), s.values, color=[GCOL[GROUP[t]] for t in s.index], width=0.7)
    for i, (t, v) in enumerate(s.items()): ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(s))); ax.set_xticklabels([nm(t) for t in s.index], rotation=40, ha="right", fontsize=9); ax.set_ylim(-0.22, 1.1); ax.axhline(0, color=DIM, linewidth=0.9)
    ax.set_ylabel("seasonal R-squared, out of sample by year"); ax.set_title("How much of a topic's year repeats", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in GCOL.items()], frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right")


def weekday(ax):
    order = WD["swing_pct"].sort_values(ascending=False).index; Z = np.array([[100 * (np.exp(WD.loc[t, f"e{k}"]) - 1) for k in range(7)] for t in order]); cm = LinearSegmentedColormap.from_list("w", [ORANGE, "#1d1d1d", BLUE])
    ax.grid(False); im = ax.imshow(Z, cmap=cm, vmin=-25, vmax=25, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(7): ax.text(j, i, f"{Z[i, j]:+.0f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(7)); ax.set_xticklabels(DAYS, fontsize=9); ax.set_yticks(range(len(order))); ax.set_yticklabels([nm(t) for t in order], fontsize=9); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Weekday effect, % against the weekly average", fontsize=12)


def holiday_lines(axes):
    yrs = range(2016, 2026); ix = {ts.date(): i for i, ts in enumerate(IDX)}; v = V["Easter"]; cols = plt.cm.cool(np.linspace(0.1, 0.9, 10)); ax0, ax1 = axes
    for c, y in zip(cols, yrs):
        e = easter(y); xs = np.arange(-30, 31); ys = [v[ix[e + timedelta(int(o))]] for o in xs]
        dt = [e + timedelta(int(o)) for o in xs]; cal = [(d - date(y, 3, 1)).days for d in dt]
        ax0.plot(cal, ys, color=c, linewidth=1.4); ax1.plot(xs, ys, color=c, linewidth=1.4)
    for ax in axes: ax.set_yscale("log"); ax.set_ylim(800, 600000); ax.set_ylabel("daily views of Easter" if ax is ax0 else "")
    ax0.set_xticks([0, 10, 20, 31, 41, 51, 61, 72]); ax0.set_xticklabels(["1 Mar", "11 Mar", "21 Mar", "1 Apr", "11 Apr", "21 Apr", "1 May", "12 May"], fontsize=8.5); ax0.set_xlim(-8, 72)
    ax0.set_title(f"Each year by calendar date: seven weeks of peaks (R² {AL.loc['Easter', 'calendar_r2']:.2f})", fontsize=11); ax1.set_title(f"By days from Easter Sunday (R² {AL.loc['Easter', 'aligned_r2']:.2f})", fontsize=11); ax1.set_xlabel("days from Easter Sunday"); ax1.set_xlim(-30, 30)
    ax0.text(0.02, 0.05, "10 years, 2016 to 2025", transform=ax0.transAxes, color=DIM, fontsize=9)


def decay(ax):
    k = np.arange(29); ix = {int(ts.strftime("%Y%m%d")): i for i, ts in enumerate(IDX)}
    for _, r in SP.iterrows():
        v = V[r["topic"]]; p = ix[int(r["peak_date"])]; ex = v[p:p + 29] - r["baseline"]
        if len(ex) < 29: continue
        e = ex / ex[0]; e = np.where(e > 0, e, np.nan); c, a, z = ("#5a5a58", 0.5, 2)
        if r["topic"] == "Eurovision_Song_Contest": c, a, z = BLUE, 0.9, 3
        if r["topic"] == "Olympic_Games": c, a, z = ORANGE, 0.9, 3
        ax.plot(k, e, color=c, alpha=a, linewidth=1.2 if z == 2 else 1.8, zorder=z)
    ax.plot(k, 0.5 ** k, color=GREEN, linestyle=":", linewidth=1.8, zorder=4); pass
    ax.axhline(0.1, color=DIM, linestyle="--", linewidth=1); ax.text(28, 0.115, "10%", color=DIM, ha="right", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(0.003, 1.2); ax.set_xlim(0, 28); ax.set_xlabel("days after the peak"); ax.set_ylabel("excess over the baseline, as a share of the peak's")
    ax.set_title("Every spike: Eurovision blue, Olympics orange,\nthe dotted line is a 1-day half-life", fontsize=11)


def halflife(ax):
    s = SP[SP["halflife"].notna()]; order = ["Eurovision_Song_Contest", "FIFA_World_Cup", "Solar_eclipse_of_April_8,_2024", "Thanksgiving", "Measles"]; rng = np.random.default_rng(3)
    for i, t in enumerate(order):
        h = s[s["topic"] == t]["halflife"].values; ax.scatter(i + rng.uniform(-0.12, 0.12, len(h)), h, color=BLUE if t != "Measles" else ORANGE, s=42, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(h)] * 2, color=INK, linewidth=2, zorder=4); ax.text(i + 0.3, np.median(h), f"{np.median(h):.1f}", color=INK, fontsize=9, va="center")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{nm(t)}\n({int((s['topic'] == t).sum())})" for t in order], fontsize=9); ax.set_ylim(0, 6.5); ax.set_xlim(-0.5, len(order) - 0.3)
    ax.set_ylabel("half-life, days (clean fits only)"); ax.set_title(f"Half-life of the {len(s)} clean fits (of {len(SP)} spikes)", fontsize=12)


def walkforward(axes):
    ax, bx = axes; order = ["Easter", "Halloween", "Christmas", "Thanksgiving", "Eurovision_Song_Contest", "Influenza", "Measles", "FIFA_World_Cup", "Olympic_Games", "Solar_eclipse_of_April_8,_2024", "Japan", "Chicago"]; x = np.arange(len(order)); w = 0.27
    ax.bar(x - w, [WF.loc[t, "mae_date_last"] for t in order], width=w * 0.92, color=GRAY, label="same date as last year")
    ax.bar(x, [WF.loc[t, "mae_date_profile"] for t in order], width=w * 0.92, color=BLUE, label="peak of the average year")
    ax.bar(x + w, [0 if np.isnan(WF.loc[t, "mae_date_rule"]) else WF.loc[t, "mae_date_rule"] for t in order], width=w * 0.92, color=GREEN, label="holiday rule + usual offset")
    ax.axhline(SS["null_date_error"], color=ORANGE, linestyle="--", linewidth=1.3); ax.text(-0.4, SS["null_date_error"] + 2, "a random date: 91 days", color=ORANGE, ha="left", fontsize=9); ax.set_ylim(0, 150)
    for xi in (0, 3): ax.text(xi + w, 3, "0", color=GREEN, ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([nm(t) for t in order], rotation=45, ha="right", fontsize=8.5); ax.set_ylabel("mean miss of the peak date, days"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left"); ax.set_title("Next year's peak date", fontsize=12)
    bx.bar(x - w / 2, [WF.loc[t, "mae_h_last"] for t in order], width=w * 1.5 * 0.92, color=ORANGE, label="last year's peak"); bx.bar(x + w / 2 + 0.03, [WF.loc[t, "mae_h_ratio"] for t in order], width=w * 1.5 * 0.92, color=BLUE, label="last year's median x usual peak ratio")
    bx.set_xticks(x); bx.set_xticklabels([nm(t) for t in order], rotation=45, ha="right", fontsize=8.5); bx.set_ylabel("mean absolute log error (0.25 is about 28% off)"); bx.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left"); bx.set_title("Next year's peak height", fontsize=12)


def displacement(axes):
    ax, bx = axes; rel = DP[DP["related"] == 1].copy(); rel["lab"] = rel["a"].map(nm) + " / " + rel["b"].map(nm); rel = rel.sort_values("r_yoy"); y = np.arange(len(rel))
    ax.hlines(y, rel["ci_lo"], rel["ci_hi"], color=GRAY, linewidth=2); ax.scatter(rel["r_yoy"], y, color=BLUE, s=48, zorder=3, label="correlation, weekly change"); ax.scatter(rel["r_partial"], y, color=GREEN, s=26, zorder=4, marker="D", label="after the common factor")
    ot = DP[(DP["related"] == 0) & (DP["evergreen_pair"] == 0)]; ax.axvline(0, color=DIM, linewidth=0.9); ax.axvspan(ot["r_yoy"].quantile(0.1), ot["r_yoy"].quantile(0.9), color="#2a2a2a", zorder=1); ax.text((ot["r_yoy"].quantile(0.1) + ot["r_yoy"].quantile(0.9)) / 2, 5.5, "middle 80%\nof the 37\nother pairs", color=DIM, fontsize=8, ha="center", va="center")
    ax.set_yticks(y); ax.set_yticklabels(rel["lab"], fontsize=8.5); ax.set_xlabel("correlation of the year-on-year change of weekly views"); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower right"); ax.set_title("Related topics: positive, not displacing", fontsize=12)
    g = AG.groupby("year")[["user", "automated", "spider"]].sum(); yrs = g.index.values; w = 0.4
    bx.bar(yrs - w / 2, 100 * g["automated"] / g["user"], width=w * 0.95, color=ORANGE, label="automated, flagged"); bx.bar(yrs + w / 2, 100 * g["spider"] / g["user"], width=w * 0.95, color=GRAY, label="spiders (crawlers)")
    bx.set_xticks(yrs); bx.set_xticklabels([str(v)[2:] if v < 2026 else "26*" for v in yrs], fontsize=9); bx.set_ylabel("% of the 'user' views, 12 topics pooled"); bx.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left"); bx.set_title("What the filter removes (flagged from 29 April 2020)", fontsize=12)


if __name__ == "__main__":
    (HERE / "charts").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 8)); g = fig.add_gridspec(1, 2, width_ratios=[1.25, 1]); a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1]); year_profiles(a, 11); t10_hist(b)
    fig.suptitle(f"Holidays repeat like clockwork ({100 * SE.loc['Halloween', 'seasonal_r2']:.0f}% of Halloween's year), and most spikes are 90% gone in 3 days", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1, 1]}); seasonality(axes[0]); weekday(axes[1]); save(fig, "01_seasonality.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True); holiday_lines(axes); save(fig, "02_alignment.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); decay(axes[0]); halflife(axes[1]); save(fig, "03_spikes.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3)); walkforward(axes); save(fig, "04_walkforward.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); displacement(axes); save(fig, "05_displacement.png")
