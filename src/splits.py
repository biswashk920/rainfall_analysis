"""Spatial and temporal splits for the rainfall project."""
import numpy as np
import pandas as pd


def distance_matrix(meta):
    """Great-circle distances (km) between stations; meta needs lat and lon columns."""
    R = 6371.0
    lat = np.radians(meta["lat"].values)
    lon = np.radians(meta["lon"].values)
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :]
         * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return pd.DataFrame(2 * R * np.arcsin(np.sqrt(a)), index=meta.index, columns=meta.index)


def make_folds(stations, dist, scheme="blocked", block_km=30):
    """One fold per held-out station. 'loso' trains on all other stations;
    'blocked' also removes every station closer than block_km to the held-out one."""
    folds = []
    for h in stations:
        if scheme == "loso":
            train = [s for s in stations if s != h]
        elif scheme == "blocked":
            train = [s for s in stations if s != h and dist.loc[h, s] >= block_km]
        else:
            raise ValueError("scheme must be 'loso' or 'blocked'")
        folds.append({"held_out": h, "train": train, "scheme": scheme})
    return folds


def fold_masks(df, fold, train_end):
    """Boolean row masks for a long table with columns station and date.
    train          : training stations, years <= train_end
    temporal_only  : training stations, years  > train_end
    spatial_only   : held-out station, years <= train_end
    headline       : held-out station, years  > train_end  (unseen site AND unseen years)"""
    early = (df["date"] <= pd.Timestamp(train_end)).values
    in_train = df["station"].isin(fold["train"]).values
    held = (df["station"] == fold["held_out"]).values
    return {"train": in_train & early, "temporal_only": in_train & ~early,
            "spatial_only": held & early, "headline": held & ~early}
