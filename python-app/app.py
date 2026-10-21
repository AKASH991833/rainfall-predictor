"""MonsoonLens - next-day rainfall prediction for Lucknow (Streamlit).
Run: streamlit run app.py
Models: XGBoost classifier + regressor on 6 years of Open-Meteo/ERA5 data.
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier, XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="MonsoonLens - Rainfall Prediction", page_icon="🌧️", layout="wide")

@st.cache_resource
def load_models():
    c = XGBClassifier(); c.load_model(os.path.join(HERE,"artifacts","clf.json"))
    r = XGBRegressor(); r.load_model(os.path.join(HERE,"artifacts","reg.json"))
    return c, r

@st.cache_data
def load_data():
    a = os.path.join(HERE,"artifacts")
    return (json.load(open(os.path.join(a,"metrics.json"))),
            pd.read_csv(os.path.join(a,"monthly.csv")),
            pd.read_csv(os.path.join(a,"recent.csv"), parse_dates=["time"]))

clf, reg = load_models()
metrics, monthly, recent = load_data()
FEATS = metrics["features"]

st.title("🌧️ MonsoonLens")
st.caption("Will it rain tomorrow in Lucknow? XGBoost trained on 2,192 days of real "
           "weather (Open-Meteo / ERA5 archive, 2019-2024).")

with st.sidebar:
    st.header("Today's conditions")
    date = st.date_input("Date", pd.Timestamp("2026-06-15"))
    tmax = st.slider("Max temperature (°C)", 5.0, 48.0, 34.0, 0.5)
    tmin = st.slider("Min temperature (°C)", 0.0, 35.0, 26.0, 0.5)
    humid = st.slider("Mean relative humidity (%)", 10.0, 100.0, 70.0, 1.0)
    press = st.slider("Surface pressure (hPa)", 985.0, 1025.0, 1002.0, 0.5)
    pdelta = st.slider("Pressure change vs yesterday (hPa)", -10.0, 10.0, -1.0, 0.5)
    cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 60.0, 1.0)
    wind = st.slider("Max wind speed (km/h)", 0.0, 80.0, 15.0, 1.0)
    gust = st.slider("Max wind gusts (km/h)", 0.0, 120.0, 30.0, 1.0)
    solar = st.slider("Solar radiation (MJ/m²)", 0.0, 32.0, 15.0, 0.5)
    precip_today = st.slider("Precipitation today (mm)", 0.0, 150.0, 0.0, 0.5)
    precip_3d = st.slider("Precipitation last 3 days (mm)", 0.0, 300.0, 2.0, 0.5)
    humid_3d = st.slider("Mean humidity last 3 days (%)", 10.0, 100.0, 70.0, 1.0)
    st.divider()
    st.subheader("Honest limits")
    st.caption(f"Next-day rain: {metrics['accuracy']*100:.0f}% accuracy, AUC {metrics['roc_auc']} "
               f"on a time-based holdout; rainy-day amount within ±{metrics['amount_mae_mm']}mm on average. "
               "Lucknow only. It learns seasonal + pressure/humidity patterns - "
               "it is not a live forecast and has no access to today's satellite data.")

def predict(row_dict):
    X = pd.DataFrame([row_dict])[FEATS]
    prob = float(clf.predict_proba(X)[0,1])
    amt = float(max(0.0, reg.predict(X)[0]))
    return prob, amt

doy = pd.Timestamp(date).dayofyear
row = {"temperature_2m_max":tmax,"temperature_2m_min":tmin,"wind_speed_10m_max":wind,
       "wind_gusts_10m_max":gust,"relative_humidity_2m_mean":humid,"surface_pressure_mean":press,
       "cloud_cover_mean":cloud,"shortwave_radiation_sum":solar,"precipitation_sum":precip_today,
       "month":pd.Timestamp(date).month,"doy_sin":np.sin(2*np.pi*doy/365.25),
       "doy_cos":np.cos(2*np.pi*doy/365.25),"temp_range":tmax-tmin,
       "precip_3d":precip_3d,"humid_3d":humid_3d,"pressure_delta":pdelta}
prob, amt = predict(row)

c1, c2, c3 = st.columns(3)
c1.metric("Rain probability tomorrow", f"{prob*100:.0f}%")
c2.metric("Expected amount (if it rains)", f"{amt:.1f} mm")
c3.metric("Verdict", "Rain likely ☔" if prob >= 0.5 else "Probably dry ☀️")

st.subheader("Lucknow's climate rhythm (2019-2024)")
m = monthly.set_index("month")
st.bar_chart(m["precip_mm"])
st.caption("Total precipitation by month across all 6 years (mm) - the monsoon pulse is unmistakable.")

st.subheader("Model vs. reality (last 6 months of holdout)")
r = recent.set_index("time")[["precip_tomorrow","prob"]].rename(
    columns={"precip_tomorrow":"actual rain (mm)","prob":"predicted probability"})
st.line_chart(r)
