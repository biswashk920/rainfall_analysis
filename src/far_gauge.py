"""Far-gauge features for notebook 05."""
import numpy as np
import pandas as pd

import splits

NB_COLS = ["nb_rel_mean", "nb_wet_frac", "nb_rel_max", "nb_rel_dm1", "nb_rel_dp1"]


def build_far_features(D, rain, train_stations, far_km=30):
    """Five features per model-table row, from training-set stations at least far_km away.
    The target station itself is never used; there is no neighbour count."""
    dist = splits.distance_matrix(D.meta)
    Rrel = rain / rain.loc[:D.train_end].mean()
    st_arr = D.valid["station"].to_numpy()
    date_pos = rain.index.get_indexer(pd.DatetimeIndex(D.dates))
    assert (date_pos >= 0).all(), "some model-table dates are not in the gauge calendar"
    out = np.full((len(D.valid), len(NB_COLS)), np.nan)
    for s in D.stations:
        elig = [u for u in train_stations if u != s and dist.loc[s, u] >= far_km]
        if not elig:
            continue
        rows_s = np.where(st_arr == s)[0]
        w = 1.0 / dist.loc[s, elig].to_numpy() ** 2
        R, raw = Rrel[elig].to_numpy(), rain[elig].to_numpy()
        ok = np.isfinite(R)
        den = (ok * w).sum(axis=1)
        safe = np.where(den > 0, den, 1.0)
        mean_rel = np.where(den > 0, (np.where(ok, R, 0.0) * w).sum(axis=1) / safe, np.nan)
        wetfrac = np.where(den > 0, (np.where(ok, (raw >= 1.0).astype(float), 0.0) * w).sum(axis=1) / safe, np.nan)
        mx = np.where(ok.any(axis=1), np.max(np.where(ok, R, -np.inf), axis=1), np.nan)
        ms = pd.Series(mean_rel)
        feats = np.column_stack([mean_rel, wetfrac, mx, ms.shift(1).to_numpy(), ms.shift(-1).to_numpy()])
        out[rows_s] = feats[date_pos[rows_s]]
    return out
