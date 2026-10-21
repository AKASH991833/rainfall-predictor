---
title: MonsoonLens
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# MonsoonLens

Next-day rainfall prediction for Lucknow (Open-Meteo ERA5).

## Run locally
```bash
cd python-app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces
Create a new Space (SDK: streamlit), upload the contents of this folder, and it runs as-is -
`requirements.txt` and `app_file: app.py` are already set. No card or paid plan needed
(free CPU tier).
