"""Train MonsoonLens on 6 years of real Lucknow weather (Open-Meteo archive).
Two models: next-day rain classifier + precipitation-amount regressor.
Artifacts: clf.json, reg.json, metrics.json, monthly.csv, recent.csv
Run: python train.py
"""
import json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = ["temperature_2m_max","temperature_2m_min","wind_speed_10m_max","wind_gusts_10m_max",
        "relative_humidity_2m_mean","surface_pressure_mean","cloud_cover_mean",
        "shortwave_radiation_sum","precipitation_sum"]
FEATS = BASE + ["month","doy_sin","doy_cos","temp_range","precip_3d","humid_3d","pressure_delta"]

def load():
    df = pd.read_csv(os.path.join(HERE,"data","lucknow_daily.csv"), parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["month"] = df.time.dt.month
    doy = df.time.dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/365.25); df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    df["temp_range"] = df.temperature_2m_max - df.temperature_2m_min
    df["precip_3d"] = df.precipitation_sum.rolling(3).sum().fillna(0)
    df["humid_3d"] = df.relative_humidity_2m_mean.rolling(3).mean().bfill()
    df["pressure_delta"] = df.surface_pressure_mean.diff().fillna(0)
    # targets: TOMORROW's weather
    df["rain_tomorrow"] = (df.precipitation_sum.shift(-1) > 1.0).astype(int)
    df["precip_tomorrow"] = df.precipitation_sum.shift(-1)
    return df.dropna().reset_index(drop=True)

def main():
    df = load()
    cut = df.time.quantile(0.8)
    tr, te = df[df.time <= cut], df[df.time > cut]
    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.06,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=42)
    clf.fit(tr[FEATS], tr.rain_tomorrow)
    p = clf.predict(te[FEATS]); prob = clf.predict_proba(te[FEATS])[:,1]
    acc = float(accuracy_score(te.rain_tomorrow, p))
    auc = float(roc_auc_score(te.rain_tomorrow, prob))
    reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.06,
                       subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=42)
    rain_tr = tr[tr.rain_tomorrow == 1]
    reg.fit(rain_tr[FEATS], rain_tr.precip_tomorrow)
    rain_te = te[te.rain_tomorrow == 1]
    mae = float(mean_absolute_error(rain_te.precip_tomorrow, reg.predict(rain_te[FEATS])))
    print(f"rain-tomorrow acc={acc:.3f} AUC={auc:.3f} | rainy-day amount MAE={mae:.1f}mm (n_test={len(te)})")

    out = os.path.join(HERE,"artifacts"); os.makedirs(out, exist_ok=True)
    clf.save_model(os.path.join(out,"clf.json")); reg.save_model(os.path.join(out,"reg.json"))
    json.dump({"accuracy":round(acc,3),"roc_auc":round(auc,3),"amount_mae_mm":round(mae,1),
               "n_days":int(len(df)),"n_test":int(len(te)),"city":"Lucknow, India (26.85N, 80.95E)",
               "source":"Open-Meteo historical archive (ERA5), 2019-2024","features":FEATS},
              open(os.path.join(out,"metrics.json"),"w"), indent=2)
    df.assign(rain=(df.precipitation_sum>1).astype(int)).groupby("month") \
      .agg(precip_mm=("precipitation_sum","sum"), rainy_days=("rain","sum")).round(1).reset_index() \
      .to_csv(os.path.join(out,"monthly.csv"), index=False)
    te.assign(prob=prob.round(3))[["time","precipitation_sum","precip_tomorrow","prob"]].tail(180) \
      .to_csv(os.path.join(out,"recent.csv"), index=False)
    print("artifacts written")

if __name__ == "__main__":
    main()
