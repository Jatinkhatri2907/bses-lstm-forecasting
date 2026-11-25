# ⚡ BSES Rajdhani – LSTM Short-Term Demand Forecaster
Weather-aware 24-hour-ahead electricity demand forecasting (168h history, LSTM encoder + future-weather branch),
with baselines, per-season evaluation, peak-load alerts and a Streamlit dashboard.

## Run locally
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free (Streamlit Community Cloud)
1. `git init && git add . && git commit -m "BSES LSTM forecaster"`
2. Create a GitHub repo, then `git remote add origin <url> && git push -u origin main`
3. Go to share.streamlit.io → New app → pick repo, branch `main`, file `app.py`
4. Advanced settings → **Python 3.11** (needed for TensorFlow) → Deploy.

## Data
- Default: simulated Delhi-style demand + weather (clearly labelled; NOT real BSES data).
- Real: upload CSV with columns `datetime,demand_mw`; Delhi weather is fetched from Open-Meteo.
