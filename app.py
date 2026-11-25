"""
BSES Rajdhani - Short-Term Electricity Demand Forecasting (LSTM + Weather)
Run:  streamlit run app.py
"""
import io, numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="BSES Rajdhani Demand Forecaster", page_icon="⚡", layout="wide")
L, H = 168, 24  # 7-day lookback, 24-hour horizon
WX = ["temperature", "humidity", "precipitation", "wind_speed", "pressure"]
TIME = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
FEATS = ["demand_mw"] + WX + TIME
FUT = ["temperature", "humidity", "precipitation", "hour_sin", "hour_cos", "is_weekend"]  # known-in-advance covariates

# ------------------------------------------------------------------ data
def synth_weather(idx, seed=1):
    r = np.random.default_rng(seed); doy = idx.dayofyear.values; h = idx.hour.values; m = idx.month.values
    t = 25 + 10 * np.cos(2 * np.pi * (doy - 160) / 365) + 5 * np.sin(2 * np.pi * (h - 9) / 24)
    t = t + pd.Series(r.normal(0, 1.2, len(idx))).rolling(24, min_periods=1).mean().values * 2
    monsoon = (m >= 7) & (m <= 9)
    hum = np.clip(55 - 1.1 * (t - 25) + 22 * monsoon + r.normal(0, 6, len(idx)), 12, 100)
    rain = np.where(monsoon & (r.random(len(idx)) < 0.12), r.exponential(3, len(idx)), 0.0)
    wind = np.clip(8 + 3 * np.sin(2 * np.pi * (h - 14) / 24) + r.normal(0, 2, len(idx)), 0, None)
    pres = 1008 - 8 * np.cos(2 * np.pi * (doy - 160) / 365) + r.normal(0, 1.5, len(idx))
    return pd.DataFrame(dict(temperature=t, humidity=hum, precipitation=rain, wind_speed=wind, pressure=pres), index=idx)

def synth_demand(w, seed=2):
    r = np.random.default_rng(seed); h = w.index.hour.values; T = w.temperature.values
    g = lambda c, s: np.exp(-0.5 * ((h - c) / s) ** 2)
    d = 1700 + 380 * g(14, 4) + 520 * g(22, 3) + 90 * np.maximum(T - 24, 0) + 25 * np.maximum(14 - T, 0)
    d += 2.0 * np.maximum(T - 28, 0) * np.maximum(w.humidity.values - 40, 0) / 10
    d *= np.where(w.index.dayofweek >= 5, 0.94, 1.0) * (1 + 0.012 * (w.index.year.values - w.index.year.min()))
    return pd.Series(d * (1 + r.normal(0, 0.02, len(d))), index=w.index, name="demand_mw")

@st.cache_data(show_spinner=False)
def open_meteo(start, end):
    p = dict(latitude=28.6139, longitude=77.2090, start_date=start, end_date=end, timezone="Asia/Kolkata",
             hourly="temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl")
    j = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=40).json()["hourly"]
    d = pd.DataFrame(j); d.index = pd.to_datetime(d.pop("time"))
    d.columns = WX; return d

def add_time(d):
    i = d.index
    for n, v, P in [("hour", i.hour, 24), ("dow", i.dayofweek, 7), ("month", i.month, 12)]:
        d[n + "_sin"] = np.sin(2 * np.pi * v / P); d[n + "_cos"] = np.cos(2 * np.pi * v / P)
    d["is_weekend"] = (i.dayofweek >= 5).astype(int); return d

@st.cache_data(show_spinner="Preparing dataset...")
def load_data(source, years, raw):
    note = ""
    if source == "Upload real load CSV" and raw:
        x = pd.read_csv(io.BytesIO(raw)); x.index = pd.to_datetime(x.iloc[:, 0]); dem = x.iloc[:, 1].astype(float)
        dem = dem.resample("h").mean().interpolate("time").rename("demand_mw")
        try: w = open_meteo(str(dem.index[0].date()), str(dem.index[-1].date())).reindex(dem.index).interpolate().ffill().bfill()
        except Exception: w = synth_weather(dem.index); note = "Open-Meteo unreachable: used simulated Delhi weather."
    else:
        idx = pd.date_range("2022-01-01", periods=int(years * 8760), freq="h")
        w = synth_weather(idx); dem = synth_demand(w)
        note = "SIMULATED Delhi-style demand (demo only, not real BSES data)."
    return add_time(pd.concat([dem, w], axis=1).dropna()), note

# ------------------------------------------------------------------ model
def season(m): return "Winter" if m in (12, 1, 2) else "Summer" if m <= 6 else "Monsoon" if m <= 9 else "Post-monsoon"

@st.cache_resource(show_spinner=False)
def train(key, _df, epochs):
    import tensorflow as tf
    from tensorflow.keras import layers as Ly
    tf.keras.utils.set_random_seed(42)
    N = len(_df); tr_end, va_end = int(N * .70), int(N * .85)
    sc = MinMaxScaler().fit(_df[FEATS].iloc[:tr_end]); F = sc.transform(_df[FEATS]).astype("float32")
    ci = [FEATS.index(c) for c in FUT]; n = N - L - H + 1
    X = swv(F, (L, F.shape[1]))[:n, 0]; C = swv(F[:, ci], (H, len(ci)))[L:L + n, 0]; Y = swv(F[:, 0], H)[L:L + n]
    k = np.arange(n); o = k + L  # forecast origin row
    m_tr, m_va, m_te = o + H <= tr_end, (o >= tr_end) & (o + H <= va_end), o >= va_end
    ip = Ly.Input((L, F.shape[1])); x = Ly.Dropout(.2)(Ly.LSTM(128, return_sequences=True)(ip))
    x = Ly.Dropout(.2)(Ly.LSTM(64)(x)); ic = Ly.Input((H, len(ci)))
    z = Ly.Concatenate()([x, Ly.Dense(32, activation="relu")(Ly.Flatten()(ic))])
    out = Ly.Dense(H)(Ly.Dense(32, activation="relu")(Ly.Dense(64, activation="relu")(z)))
    model = tf.keras.Model([ip, ic], out); model.compile("adam", "mse", metrics=["mae"])
    hist = model.fit([X[m_tr], C[m_tr]], Y[m_tr], validation_data=([X[m_va], C[m_va]], Y[m_va]), epochs=epochs,
                     batch_size=128, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    inv = lambda a: (a - sc.min_[0]) / sc.scale_[0]
    pred = inv(model.predict([X[m_te], C[m_te]], verbose=0)); act = inv(Y[m_te]); ot = o[m_te]
    D = _df.demand_mw.values; W = swv(D, H)
    base = {"Persistence (yesterday)": np.array([W[i - 24] for i in ot]), "Persistence (last week)": np.array([W[i - 168] for i in ot])}
    return dict(model=model, sc=sc, hist=hist.history, pred=pred, act=act, origins=_df.index[ot], base=base,
                peak=float(np.percentile(D[:tr_end], 95)), params=model.count_params())

def metrics(a, p):
    a, p = a.ravel(), p.ravel()
    return dict(MAE=mean_absolute_error(a, p), RMSE=float(np.sqrt(mean_squared_error(a, p))),
                MAPE=float(np.mean(np.abs((a - p) / a)) * 100), R2=r2_score(a, p))

# ------------------------------------------------------------------ UI
st.title("⚡ BSES Rajdhani – LSTM Short-Term Demand Forecaster")
st.caption("168-hour history + weather → next-24-hour demand. Weather-aware: the model also receives the 24-hour weather/calendar outlook.")
with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["Simulated Delhi demand (demo)", "Upload real load CSV"])
    raw = None
    if source.startswith("Upload"):
        up = st.file_uploader("CSV: datetime, demand_mw (hourly or finer)", type="csv"); raw = up.getvalue() if up else None
        st.info("Weather is auto-fetched from Open-Meteo for Delhi.")
    years = st.slider("Simulated years", 1.0, 4.0, 2.0, 0.5, disabled=source.startswith("Upload"))
    epochs = st.slider("Max epochs", 3, 40, 15)
    go_btn = st.button("🚀 Train / Load model", type="primary", use_container_width=True)
if source.startswith("Upload") and raw is None: st.warning("Upload a CSV or switch to the simulated demo."); st.stop()
df, note = load_data(source if source.startswith("Upload") else "sim", years, raw)
(st.warning if "SIMULATED" in note or "unreachable" in note else st.success)(note or "Real load data loaded.")
if go_btn: st.session_state["go"] = True
if not st.session_state.get("go"): st.info("Press **Train / Load model** in the sidebar."); st.stop()
with st.spinner("Training LSTM (first run only, ~1-3 min)..."):
    R = train(f"{source}-{years}-{epochs}-{len(df)}", df, epochs)

pred, act, org = R["pred"], R["act"], R["origins"]
mt = metrics(act, pred); mb = {k: metrics(act, v) for k, v in R["base"].items()}
tabs = st.tabs(["📈 Forecast", "📊 Evaluation", "🔍 Data insights", "🏗 Model"])

with tabs[0]:
    i = st.slider("Forecast origin (test period)", 0, len(org) - 1, len(org) // 2)
    ts = pd.date_range(org[i], periods=H, freq="h"); p, a = pred[i], act[i]
    f = go.Figure([go.Scatter(x=ts, y=a, name="Actual", line=dict(width=3)), go.Scatter(x=ts, y=p, name="LSTM forecast", line=dict(dash="dash", width=3))])
    f.add_hline(y=R["peak"], line_color="red", annotation_text="Peak threshold (P95)"); f.update_layout(yaxis_title="Demand (MW)", height=420)
    st.plotly_chart(f, use_container_width=True)
    c = st.columns(4); c[0].metric("Predicted peak (MW)", f"{p.max():.0f}"); c[1].metric("Peak hour", ts[p.argmax()].strftime("%a %H:%M"))
    c[2].metric("Window MAPE", f"{np.mean(abs(a - p) / a) * 100:.2f}%"); c[3].metric("Hours above threshold", int((p > R['peak']).sum()))
    if (p > R["peak"]).any(): st.error(f"⚠️ Peak-load alert: {int((p > R['peak']).sum())} hour(s) forecast above {R['peak']:.0f} MW – consider load-shedding/purchase planning.")
    else: st.success("✅ No peak-load alert in this 24-hour window.")
    st.download_button("Download forecast CSV", pd.DataFrame({"datetime": ts, "forecast_mw": p, "actual_mw": a}).to_csv(index=False), "forecast_24h.csv")

with tabs[1]:
    tb = pd.DataFrame({"LSTM": mt, **mb}).T.round(3); st.dataframe(tb, use_container_width=True)
    best = min(mb.values(), key=lambda m: m["MAE"])["MAE"]; st.metric("MAE improvement vs best baseline", f"{(1 - mt['MAE'] / best) * 100:.1f}%")
    h_mae = np.abs(act - pred).mean(0); st.plotly_chart(go.Figure(go.Bar(x=list(range(1, H + 1)), y=h_mae)).update_layout(title="MAE by forecast horizon (hours ahead)", xaxis_title="Hours ahead", yaxis_title="MAE (MW)"), use_container_width=True)
    S = np.array([season(m) for m in org.month]); rows = {s: metrics(act[S == s], pred[S == s]) for s in np.unique(S)}
    st.subheader("Performance by season (test period)"); st.dataframe(pd.DataFrame(rows).T.round(3), use_container_width=True)
    e = go.Figure([go.Scatter(y=R["hist"]["loss"], name="Train loss"), go.Scatter(y=R["hist"]["val_loss"], name="Validation loss")]); st.plotly_chart(e.update_layout(title="Training curve (MSE)"), use_container_width=True)

with tabs[2]:
    d = df.copy(); d["h"], d["w"] = d.index.hour, d.index.dayofweek
    st.plotly_chart(go.Figure(go.Heatmap(z=d.pivot_table("demand_mw", "w", "h").values, x=list(range(24)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], colorscale="YlOrRd")).update_layout(title="Average demand: weekday × hour"), use_container_width=True)
    cr = df[["demand_mw"] + WX].corr(); st.plotly_chart(go.Figure(go.Heatmap(z=cr.values, x=cr.columns, y=cr.columns, zmin=-1, zmax=1, colorscale="RdBu", text=cr.round(2).values, texttemplate="%{text}")).update_layout(title="Demand–weather correlation"), use_container_width=True)
    st.plotly_chart(go.Figure(go.Scatter(x=df.temperature, y=df.demand_mw, mode="markers", marker=dict(size=3, opacity=.3))).update_layout(title="Temperature vs demand", xaxis_title="°C", yaxis_title="MW"), use_container_width=True)

with tabs[3]:
    st.markdown(f"""**Architecture** (functional Keras, {R['params']:,} parameters)
- Encoder: Input 168h × {len(FEATS)} features → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
- Future-covariate branch: next-24h temperature, humidity, rain, hour & weekend flags → Dense(32)
- Fusion: concat → Dense(64) → Dense(32) → Dense(24) hourly outputs

**Methodology:** chronological 70/15/15 split, scalers fit on training data only, no leakage across boundaries, EarlyStopping, comparison with persistence baselines.

**Honesty note:** the simulated mode is a pipeline demonstration. For genuine BSES results, upload real hourly load data in the sidebar. In production, the future-weather input should come from a weather forecast, not observations.""")
