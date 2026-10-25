"""Shared code for notebook 05 (written by 05_xgboost_model, Cell 5)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import metrics
import splits

SEASONS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
CAL = ["doy_sin", "doy_cos"]
CHIRPS_F = ["chirps_c_dm1", "chirps_c_d0", "chirps_c_dp1", "chirps_blend_c",
            "chirps_blend_mean9", "chirps_blend_max9", "chirps_blend_std9",
            "chirps_blend_mean9_prev", "chirps_blend_mean9_sum3"]
ERA_P = ["era_p_blend", "era_p_dm1", "era_p_d0", "era_p_blend_prev", "era_p_blend_sum3"]
ERA_ATM = ["anom_era_t2m_c", "anom_era_dtr_c", "anom_era_dewdep_c", "anom_era_rh_pct",
           "anom_era_u10", "anom_era_v10", "anom_era_wspd", "anom_era_sp_hpa",
           "anom_era_soil_w1", "era_dsp_hpa"]
ALL_FEATS = CAL + CHIRPS_F + ERA_P + ERA_ATM

BASE_PARAMS = dict(learning_rate=0.05, max_depth=4, min_child_weight=30, subsample=0.8,
                   colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", n_jobs=-1,
                   random_state=0, objective="reg:tweedie", tweedie_variance_power=1.5)

QGRID = np.concatenate([np.linspace(0, 0.99, 100), [0.995, 0.999, 1.0]])


def fit_qm(src, tgt):
    ok = np.isfinite(src) & np.isfinite(tgt)
    s, g = src[ok], tgt[ok]
    if len(s) < 200:
        return None
    return np.quantile(s, QGRID) + np.arange(len(QGRID)) * 1e-9, np.quantile(g, QGRID)


def apply_qm(x, fit):
    sq, gq = fit
    out = np.interp(x, sq, gq)
    hi = x > sq[-1]
    out[hi] = gq[-1] * x[hi] / sq[-1]
    return out


def calibrate(src_val, tgt_val, season_val, x_te, season_te):
    """Season-by-season quantile mapping fitted on (src_val -> tgt_val), applied to x_te."""
    out = np.array(x_te, dtype=float, copy=True)
    for s in SEASONS:
        fit = fit_qm(src_val[season_val == s], tgt_val[season_val == s])
        t = season_te == s
        if fit is not None and t.any():
            out[t] = apply_qm(x_te[t], fit)
    return out


def multiscale(d, o, sm):
    """r5: correlation of non-overlapping 5-day sums. r_mon: correlation of monthly
    totals after removing each calendar month's mean and spread."""
    out = {"r5": np.nan, "r_mon": np.nan}
    df = pd.DataFrame({"obs": o, "sim": sm}, index=pd.DatetimeIndex(d)).dropna().sort_index()
    if len(df) < 100:
        return out
    g5 = df.resample("5D").agg(["sum", "count"])
    ok = g5[("obs", "count")] == 5
    if ok.sum() > 20:
        out["r5"] = float(g5.loc[ok, ("obs", "sum")].corr(g5.loc[ok, ("sim", "sum")]))
    gm = df.groupby([df.index.year, df.index.month]).agg(["sum", "count"])
    gm = gm[gm[("obs", "count")] >= 25]
    mo = pd.DataFrame({"obs": gm[("obs", "sum")], "sim": gm[("sim", "sum")]})
    mo["m"] = mo.index.get_level_values(1)

    def z(col):
        grp = mo.groupby("m")[col]
        return ((mo[col] - grp.transform("mean")) / grp.transform("std")).replace([np.inf, -np.inf], np.nan)

    if len(mo) > 20:
        out["r_mon"] = float(z("obs").corr(z("sim")))
    return out


class Data:
    """Model table, features normalised by each location's own satellite climatology
    (1983-2010), fold masks for both schemes, and the validation block (2006-2010)."""

    def __init__(self, root):
        root = Path(root)
        self.proc = root / "data" / "processed"
        self.sat = root / "data" / "satellite"
        self.out = root / "outputs"
        with open(self.proc / "eda_decisions.json") as f:
            self.dec = json.load(f)
        self.train_end = self.dec["temporal_split"]["train"][1]
        self.thr = self.dec["extreme_thresholds_mm"]
        self.meta = pd.read_csv(self.proc / "station_meta.csv", index_col="station")
        self.stations = list(self.meta.index)
        dist = splits.distance_matrix(self.meta)
        self.folds = {
            "blocked": splits.make_folds(self.stations, dist, "blocked",
                                         self.dec["spatial_split"]["block_km"]),
            "loso": splits.make_folds(self.stations, dist, "loso"),
        }
        base = pd.read_parquet(self.proc / "model_base.parquet")
        v = (base.dropna(subset=["rain", "chirps_blend_c", "era_p_blend"])
                 .sort_values(["station", "date"]).reset_index(drop=True))
        missing = [c for c in ALL_FEATS if c not in v.columns]
        assert not missing, f"missing columns: {missing}"
        self.valid = v
        self.obs = v["rain"].to_numpy()
        self.dates = v["date"].to_numpy()
        self.seas = v["season"].to_numpy()
        self.chirps = v["chirps_blend_c"].to_numpy()
        self.era_rain = v["era_p_blend"].to_numpy()
        self.early_fit = (v["date"] <= pd.Timestamp("2005-12-31")).to_numpy()
        self.val_block = ((v["date"] >= pd.Timestamp("2006-01-01"))
                          & (v["date"] <= pd.Timestamp("2010-12-31"))).to_numpy()
        sat = pd.read_parquet(self.sat / "satellite_features_model.parquet")
        tr = sat[sat["date"] <= self.train_end]
        self.cm = v["station"].map(tr.groupby("station")["chirps_blend_c"].mean()).to_numpy()
        self.emv = v["station"].map(tr.groupby("station")["era_p_blend"].mean()).to_numpy()
        self.feats = ALL_FEATS
        X = v[self.feats].to_numpy(dtype=float).copy()
        for c in CHIRPS_F:
            X[:, self.feats.index(c)] /= self.cm
        for c in ERA_P:
            X[:, self.feats.index(c)] /= self.emv
        self.X = X
        self.yrel = self.obs / self.cm
        self.masks = {(s, fo["held_out"]): splits.fold_masks(v, fo, self.train_end)
                      for s, fl in self.folds.items() for fo in fl}


def fit_predict(D, m, cols, params, pred_masks):
    """Train on the fold's training stations (fit 1983-2005, early stopping 2006-2010,
    refit on 1983-2010). Returns validation predictions (relative scale), validation
    target and seasons, predictions for each mask in pred_masks (relative scale) and
    the best number of trees."""
    idx = [D.feats.index(c) for c in cols]
    i_fit = m["train"] & D.early_fit
    i_val = m["train"] & D.val_block
    i_all = m["train"]
    mdl = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=50, **params)
    mdl.fit(D.X[i_fit][:, idx], D.yrel[i_fit],
            eval_set=[(D.X[i_val][:, idx], D.yrel[i_val])], verbose=False)
    best = int(mdl.best_iteration) + 1
    pv = mdl.predict(D.X[i_val][:, idx])
    fin = xgb.XGBRegressor(n_estimators=max(50, int(best * 1.15)), **params)
    fin.fit(D.X[i_all][:, idx], D.yrel[i_all], verbose=False)
    outs = {k: fin.predict(D.X[mk][:, idx]) for k, mk in pred_masks.items()}
    return pv, D.yrel[i_val], D.seas[i_val], outs, best


def score(D, h, i, sim):
    sc = metrics.score_all(D.dates[i], D.obs[i], sim, D.thr[h])
    sc["alpha"] = float(np.std(sim) / np.std(D.obs[i]))
    sc.update(multiscale(D.dates[i], D.obs[i], sim))
    return sc
