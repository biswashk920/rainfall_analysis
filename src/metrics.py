"""Shared scoring functions for the rainfall project."""
import numpy as np
import pandas as pd

WET = 1.0


def _pair(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def pearson_r(obs, sim):
    o, s = _pair(obs, sim)
    if len(o) < 3 or o.std() == 0 or s.std() == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def rmse(obs, sim):
    o, s = _pair(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan


def mae(obs, sim):
    o, s = _pair(obs, sim)
    return float(np.mean(np.abs(s - o))) if len(o) else np.nan


def pbias(obs, sim):
    """Percent bias of the total: 100 * (sum(sim) - sum(obs)) / sum(obs)."""
    o, s = _pair(obs, sim)
    return float(100 * (s.sum() - o.sum()) / o.sum()) if len(o) and o.sum() > 0 else np.nan


def nse(obs, sim):
    o, s = _pair(obs, sim)
    den = ((o - o.mean()) ** 2).sum() if len(o) else 0
    return float(1 - ((s - o) ** 2).sum() / den) if den > 0 else np.nan


def kge(obs, sim):
    """Kling-Gupta efficiency (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    o, s = _pair(obs, sim)
    if len(o) < 3 or o.std() == 0 or o.mean() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def categorical(obs, sim, thr=WET):
    """Detection of days at or above thr: POD, FAR, CSI and frequency bias."""
    o, s = _pair(obs, sim)
    ob, sb = o >= thr, s >= thr
    hits = int((ob & sb).sum())
    miss = int((ob & ~sb).sum())
    fa = int((~ob & sb).sum())
    return {
        "pod": hits / (hits + miss) if hits + miss else np.nan,
        "far": fa / (hits + fa) if hits + fa else np.nan,
        "csi": hits / (hits + miss + fa) if hits + miss + fa else np.nan,
        "freq_bias": (hits + fa) / (hits + miss) if hits + miss else np.nan,
    }


def wet_quantile_ratio(obs, sim, q, thr=WET, min_days=50):
    """Simulated wet-day quantile divided by the observed one."""
    o, s = _pair(obs, sim)
    ow, sw = o[o >= thr], s[s >= thr]
    if len(ow) < min_days or len(sw) < min_days:
        return np.nan
    return float(np.quantile(sw, q) / np.quantile(ow, q))


def rx1day(dates, obs, sim, min_days=300):
    """Annual maximum 1-day rainfall: ratio of means (sim/obs) and correlation across years."""
    df = pd.DataFrame({"date": pd.to_datetime(dates),
                       "obs": np.asarray(obs, dtype=float),
                       "sim": np.asarray(sim, dtype=float)}).dropna()
    empty = {"rx1_years": 0, "rx1_ratio": np.nan, "rx1_r": np.nan}
    if df.empty:
        return empty
    g = df.groupby(df["date"].dt.year)
    n = g.size()
    mx = g[["obs", "sim"]].max()[n >= min_days]
    if len(mx) < 3:
        return empty
    return {"rx1_years": int(len(mx)),
            "rx1_ratio": float(mx["sim"].mean() / mx["obs"].mean()),
            "rx1_r": float(mx["obs"].corr(mx["sim"])) if len(mx) >= 5 else np.nan}


def score_all(dates, obs, sim, thr=None, fixed_extreme=50.0, wet=WET):
    """All scores for one station. thr = {'P95_wet': x, 'P99_wet': y} (station-specific)."""
    o, s = _pair(obs, sim)
    if len(o) == 0:
        return {"n": 0}
    out = {"n": int(len(o)), "mean_obs": float(o.mean()), "mean_sim": float(s.mean()),
           "r": pearson_r(o, s), "kge": kge(o, s), "nse": nse(o, s),
           "rmse": rmse(o, s), "mae": mae(o, s), "pbias": pbias(o, s),
           "wet_freq_obs": float(100 * (o >= wet).mean()),
           "wet_freq_sim": float(100 * (s >= wet).mean())}
    for k, v in categorical(o, s, wet).items():
        out["wet_" + k] = v
    out["q95_ratio"] = wet_quantile_ratio(o, s, 0.95, wet)
    out["q99_ratio"] = wet_quantile_ratio(o, s, 0.99, wet)
    out.update(rx1day(dates, obs, sim))
    levels = [("p95", thr["P95_wet"]), ("p99", thr["P99_wet"])] if thr else []
    levels.append(("fix", fixed_extreme))
    for name, val in levels:
        for k, v in categorical(o, s, val).items():
            out[name + "_" + k] = v
    return out


def score_table(df, thresholds=None, fixed_extreme=50.0):
    """df needs columns date, station, obs, sim. Returns one row of scores per station."""
    rows = {}
    for st, d in df.groupby("station"):
        thr = (thresholds or {}).get(st)
        rows[st] = score_all(d["date"].values, d["obs"].values, d["sim"].values,
                             thr, fixed_extreme)
    return pd.DataFrame(rows).T
