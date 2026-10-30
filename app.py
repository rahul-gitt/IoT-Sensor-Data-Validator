import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="NEXUS | IoT Intelligence", page_icon="◈", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "iot_model_bundle.pkl"
MODEL_PATH = BASE_DIR / "isolation_forest_model.pkl"
DATA_PATH = BASE_DIR / "iot_validated_data.csv"
DEVICE_IDS = [
    "00:0f:00:70:91:0a",
    "1c:bf:ce:15:ec:4d",
    "b8:27:eb:bf:9d:51",
]
SENSOR_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
FALLBACK_FEATURES = [
    "co", "humidity", "light", "lpg", "motion", "smoke", "temp",
    "hour", "day_of_week",
    "device_00:0f:00:70:91:0a", "device_1c:bf:ce:15:ec:4d",
    "device_b8:27:eb:bf:9d:51",
]

# ===================== PREMIUM NEXUS UI =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: radial-gradient(ellipse at top left,#123047 0%,#07111d 48%,#060d16 100%); color:#e8f2fa; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1a29,#07111d); border-right:1px solid #20384b; }
.block-container { max-width:1500px; padding-top:1.5rem; }
.hero { padding:30px; border-radius:22px; border:1px solid #24445a; background:linear-gradient(120deg,#10283b,#091725); box-shadow:0 15px 50px #0004; }
.eyebrow { color:#55e6e0; font-size:12px; letter-spacing:3px; font-weight:700; }
.metric { background:linear-gradient(145deg,#102337,#0a1927); border:1px solid #20384b; padding:20px; border-radius:16px; min-height:125px; transition:.25s; }
.metric:hover { transform:translateY(-3px); border-color:#55e6e0; box-shadow:0 8px 30px #55e6e018; }
.metric-label { color:#9bb3c7; font-size:12px; letter-spacing:1px; }
.metric-value { font-size:32px; font-weight:800; color:#f1f8ff; margin-top:8px; }
.metric-foot { color:#7893a8; font-size:12px; }
.section { margin:28px 0 12px; padding-left:12px; border-left:4px solid #55e6e0; font-size:19px; font-weight:800; }
.pill { color:#7ff3d0; background:#0d302f; border:1px solid #275b59; border-radius:30px; padding:7px 12px; font-size:11px; }
div[data-testid="stFileUploader"] section { background:#0a1a29; border:1px dashed #3c687c; border-radius:12px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; background:#10343e; border:1px solid #3b737b; color:white; font-weight:700; }
[data-testid="stForm"] { border:1px solid #20384b; border-radius:16px; padding:18px; background:#091725; }
</style>
""", unsafe_allow_html=True)

# ===================== MODEL LOADING =====================
@st.cache_resource
def load_model_bundle():
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle["model"], bundle.get("feature_columns", FALLBACK_FEATURES)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), FALLBACK_FEATURES
    return None, FALLBACK_FEATURES

model, FEATURE_COLUMNS = load_model_bundle()


def to_binary(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return (series.astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            .fillna(0).astype(int))


def add_time_features(df):
    df = df.copy()
    if "ts" in df.columns:
        if pd.api.types.is_numeric_dtype(df["ts"]):
            df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
        else:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["hour"] = df["ts"].dt.hour.fillna(datetime.now().hour).astype(int)
        df["day_of_week"] = df["ts"].dt.dayofweek.fillna(datetime.now().weekday()).astype(int)
    else:
        df["hour"] = datetime.now().hour
        df["day_of_week"] = datetime.now().weekday()
    return df


def build_features(raw, feature_columns=FEATURE_COLUMNS):
    """Create the same ordered feature matrix used when the model was trained."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = add_time_features(df)
    for col in ["co", "humidity", "lpg", "smoke", "temp"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["light", "motion"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = to_binary(df[col])

    # One-hot encode known devices; unseen/missing device becomes all-zero device flags.
    if "device" in df.columns:
        device_values = df["device"].astype(str)
    else:
        device_values = pd.Series([""] * len(df), index=df.index)
    for device_id in DEVICE_IDS:
        col = f"device_{device_id}"
        df[col] = (device_values == device_id).astype(int)

    # Model trained with 12 features (no calendar day-of-month feature).
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    X = df[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    return X, df


def rule_validation(df):
    invalid = (
        (df["humidity"] < 0) | (df["humidity"] > 100) |
        (df["temp"] < 0) | (df["temp"] > 50) |
        (df["co"] < 0) | (df["lpg"] < 0) | (df["smoke"] < 0)
    )
    return invalid.fillna(True)


def validate_readings(raw):
    if model is None:
        raise FileNotFoundError("Model bundle not found. Keep iot_model_bundle.pkl beside app.py.")
    X, enriched = build_features(raw)
    numeric_required = ["co", "humidity", "lpg", "smoke", "temp"]
    bad_rows = enriched[numeric_required].isna().any(axis=1)
    X = X.fillna(0)
    predictions = model.predict(X)
    scores = model.decision_function(X)
    enriched["anomaly"] = np.where(predictions == -1, "Anomaly", "Normal")
    enriched["anomaly_score"] = scores
    enriched["rule_invalid"] = rule_validation(enriched)
    enriched["validation_result"] = "Valid"
    enriched.loc[enriched["anomaly"] == "Anomaly", "validation_result"] = "Suspicious"
    enriched.loc[enriched["rule_invalid"] | bad_rows, "validation_result"] = "Invalid"
    return enriched


def show_metric(col, label, value, foot, color):
    col.markdown(f"""<div class="metric"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-foot">{foot}</div>
    <div style="margin-top:14px;width:34px;height:3px;border-radius:5px;background:{color}"></div></div>""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown('<div class="eyebrow">◈ NEXUS INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Control Room")
    st.caption("IoT sensor monitoring system")
    uploaded = st.file_uploader("Upload sensor CSV for batch validation", type=["csv"])
    st.divider()
    st.markdown("### Engine status")
    if model is not None:
        st.success("● Trained model loaded")
        st.caption("Isolation Forest + Rule Validation")
    else:
        st.error("Model bundle missing")
        st.caption("Place iot_model_bundle.pkl beside app.py")

# ===================== LOAD HISTORICAL DATA =====================
try:
    if uploaded is not None:
        raw_uploaded = pd.read_csv(uploaded)
        df = validate_readings(raw_uploaded)
        source = f"Batch upload · {uploaded.name}"
    elif DATA_PATH.exists():
        historical = pd.read_csv(DATA_PATH)
        # Keep historical dataset labels for dashboard, but uploaded CSVs are always re-scored.
        if "validation_result" in historical.columns and "anomaly" in historical.columns:
            df = historical.copy()
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        else:
            df = validate_readings(historical)
        source = DATA_PATH.name
    else:
        df = pd.DataFrame()
        source = "No historical CSV loaded"
except Exception as error:
    st.error(f"Could not process data: {error}")
    df = pd.DataFrame()
    source = "Upload error"

# ===================== HERO =====================
st.markdown(f"""<div class="hero"><div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">
<span class="eyebrow">IOT / DATA QUALITY OBSERVABILITY</span><span class="pill">● NEXUS ENGINE</span></div>
<h1 style="font-size:42px;line-height:1.15">Sensor intelligence.<br><span style="color:#55e6e0">Without the guesswork.</span></h1>
<p style="color:#a7bdcf">Monitor sensor integrity, unusual readings and device behavior using machine learning.</p>
<p class="eyebrow">SOURCE: {source} &nbsp; | &nbsp; RECORDS: {len(df):,} &nbsp; | &nbsp; UPDATED: {datetime.now().strftime('%d %b %Y, %H:%M')}</p></div>""", unsafe_allow_html=True)

# ===================== LIVE SENSOR INPUT =====================
st.markdown('<div class="section">Check a sensor reading</div>', unsafe_allow_html=True)
st.caption("Choose your device, enter the seven readings, and click the button. Time details are added automatically.")

with st.form("manual_sensor_form"):
    device = st.selectbox(
        "Device",
        DEVICE_IDS,
        format_func=lambda x: f"Device {DEVICE_IDS.index(x) + 1}"
    )

    st.markdown("**Enter sensor values**")
    left, middle, right = st.columns(3)
    with left:
        temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, value=25.0, step=0.5, help="Project rule limit: 0–50°C.")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        light = st.selectbox("Is the light on?", [False, True], format_func=lambda x: "Yes" if x else "No")
    with middle:
        co = st.number_input("CO level", min_value=0.0, value=0.0046, format="%.6f", help="Enter a value using the same scale as your dataset.")
        lpg = st.number_input("LPG level", min_value=0.0, value=0.0072, format="%.6f", help="Enter a value using the same scale as your dataset.")
    with right:
        smoke = st.number_input("Smoke level", min_value=0.0, value=0.0193, format="%.6f", help="Enter a value using the same scale as your dataset.")
        motion = st.selectbox("Is motion detected?", [False, True], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Check reading", use_container_width=True)

if submitted:
    one = pd.DataFrame([{
        "device": device,
        "co": co,
        "humidity": humidity,
        "light": light,
        "lpg": lpg,
        "motion": motion,
        "smoke": smoke,
        "temp": temp,
        "ts": datetime.now()
    }])
    try:
        result = validate_readings(one).iloc[0]
        st.markdown('<div class="section">Result</div>', unsafe_allow_html=True)
        result_status = result["validation_result"]
        r1, r2, r3 = st.columns(3)
        with r1:
            if result_status == "Valid":
                st.success("VALID — passed the configured checks")
            elif result_status == "Suspicious":
                st.warning("SUSPICIOUS — the model found an unusual pattern")
            else:
                st.error("INVALID — check the sensor values and project limits")
        with r2:
            st.metric("Model result", result["anomaly"])
        with r3:
            st.metric("Anomaly score", f"{result['anomaly_score']:.5f}")

        if result["anomaly"] == "Anomaly":
            st.warning("The reading differs from patterns learned by the model. This does not prove the sensor is faulty.")
        else:
            st.info("The model did not flag this reading as an anomaly. Rule checks are shown separately above.")
        st.caption("Temperature range 0–50°C is a configurable rule for this college project, not a universal safety limit.")
    except Exception as error:
        st.error(f"Could not check this reading: {error}")

# ===================== DASHBOARD ANALYTICS =====================
if df.empty:
    st.info("Upload a sensor CSV from the sidebar or place iot_validated_data.csv beside app.py to view analytics.")
    st.stop()

required_result_cols = ["validation_result", "anomaly"]
if any(c not in df.columns for c in required_result_cols):
    st.warning("Dataset is missing validation labels; upload a raw sensor CSV to validate it.")
    st.stop()

counts = df["validation_result"].value_counts()
total = len(df)
valid = int(counts.get("Valid", 0))
suspicious = int(counts.get("Suspicious", 0))
invalid = int(counts.get("Invalid", 0))

c1, c2, c3, c4 = st.columns(4)
show_metric(c1, "TOTAL READINGS", f"{total:,}", "Records processed", "#55e6e0")
show_metric(c2, "VALID", f"{valid:,}", f"{valid/total*100 if total else 0:.2f}% of readings", "#64e6a6")
show_metric(c3, "SUSPICIOUS", f"{suspicious:,}", "Flagged by ML model", "#ffca72")
show_metric(c4, "RULE INVALID", f"{invalid:,}", "Outside configured limits", "#ff6b7f")

st.markdown('<div class="section">Explore the signal</div>', unsafe_allow_html=True)
f1, f2 = st.columns(2)
with f1:
    status = st.selectbox("Validation status", ["All statuses", "Valid", "Suspicious", "Invalid"])
with f2:
    sensor = st.selectbox("Sensor trend", ["temp", "humidity", "co", "lpg", "smoke"])
view = df.copy()
if status != "All statuses":
    view = view[view["validation_result"] == status]

st.markdown('<div class="section">Monitoring overview</div>', unsafe_allow_html=True)
left, right = st.columns([0.9, 1.4])
with left:
    chart_counts = df["validation_result"].value_counts().reindex(["Valid", "Suspicious", "Invalid"], fill_value=0).rename_axis("Status").reset_index(name="Readings")
    fig = px.pie(chart_counts, names="Status", values="Readings", hole=0.72, color="Status",
                 color_discrete_map={"Valid":"#55e6e0", "Suspicious":"#ffca72", "Invalid":"#ff6b7f"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#b7cad9", height=320, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h",y=-0.1))
    fig.add_annotation(text=f"<b>{total:,}</b><br>READINGS", x=0.5, y=0.5, showarrow=False, font=dict(size=18,color="#eef8ff"))
    st.plotly_chart(fig, use_container_width=True)
with right:
    chart = view.copy()
    if "ts" in chart.columns:
        chart["ts"] = pd.to_datetime(chart["ts"], errors="coerce")
        chart = chart.dropna(subset=["ts"]).sort_values("ts")
    if len(chart) and sensor in chart.columns:
        if len(chart) > 2500:
            chart = chart.iloc[np.linspace(0, len(chart)-1, 2500, dtype=int)]
        fig = px.line(chart, x="ts" if "ts" in chart.columns else chart.index, y=sensor, color="validation_result",
                      color_discrete_map={"Valid":"#55e6e0", "Suspicious":"#ffca72", "Invalid":"#ff6b7f"}, labels={sensor:sensor.upper(),"ts":"Timestamp"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#b7cad9", height=320, margin=dict(l=10,r=10,t=35,b=10), title=f"{sensor.upper()} · READING TREND", xaxis=dict(gridcolor="#20384b"), yaxis=dict(gridcolor="#20384b"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamped readings available for this filter.")

st.markdown('<div class="section">Sensor behavior</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    sensors = [c for c in ["co","humidity","lpg","smoke","temp"] if c in view.columns]
    if sensors and len(view):
        melted = view[sensors].melt(var_name="Sensor", value_name="Reading")
        if len(melted) > 12000:
            melted = melted.sample(12000, random_state=42)
        fig = px.box(melted, x="Sensor", y="Reading", color="Sensor", points=False,
                     color_discrete_sequence=["#55e6e0","#6aa9ff","#b59cff","#ffca72","#64e6a6"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#b7cad9", height=320, showlegend=False, title="Sensor value distribution", xaxis=dict(gridcolor="#20384b"), yaxis=dict(gridcolor="#20384b"))
        st.plotly_chart(fig, use_container_width=True)
with right:
    if "ts" in view.columns:
        timeline = view.copy()
        timeline["ts"] = pd.to_datetime(timeline["ts"], errors="coerce")
        timeline = timeline.dropna(subset=["ts"])
        if len(timeline):
            timeline["Time"] = timeline["ts"].dt.floor("h")
            timeline = timeline.groupby(["Time","validation_result"]).size().reset_index(name="Readings")
            fig = px.area(timeline, x="Time", y="Readings", color="validation_result", color_discrete_map={"Valid":"#55e6e0","Suspicious":"#ffca72","Invalid":"#ff6b7f"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#b7cad9", height=320, title="Validation activity over time", xaxis=dict(gridcolor="#20384b"), yaxis=dict(gridcolor="#20384b"))
            st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section">Advanced anomaly analysis</div>', unsafe_allow_html=True)
anomaly_df = df[df["anomaly"].astype(str).str.lower() == "anomaly"].copy()
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Anomalies", f"{len(anomaly_df):,}")
    if "anomaly_score" in anomaly_df.columns:
        st.metric("Most Suspicious Score", f"{anomaly_df['anomaly_score'].min():.5f}" if len(anomaly_df) else "N/A")
with col2:
    if "device" in anomaly_df.columns and len(anomaly_df):
        device_counts = anomaly_df["device"].value_counts().head(10).rename_axis("Device").reset_index(name="Anomalies")
        fig = px.bar(device_counts, x="Device", y="Anomalies", title="Devices with Most Anomalies", color="Anomalies", color_continuous_scale="Teal")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#b7cad9")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Device-wise chart is available when the dataset includes its original device column.")
st.caption("More negative Isolation Forest scores generally indicate more unusual observations. An anomaly is not necessarily a confirmed sensor fault.")

st.markdown('<div class="section">Flagged readings · Investigation queue</div>', unsafe_allow_html=True)
flags = view[view["validation_result"].isin(["Suspicious","Invalid"])].copy()
show_cols = [c for c in ["ts","device","validation_result","anomaly_score","co","humidity","lpg","smoke","temp","light","motion"] if c in flags.columns]
if flags.empty:
    st.info("No flagged readings for this filter.")
else:
    if "anomaly_score" in flags.columns:
        flags = flags.sort_values("anomaly_score")
    st.dataframe(flags[show_cols].head(500), use_container_width=True, height=320, hide_index=True)
    st.caption(f"Showing up to 500 of {len(flags):,} flagged readings.")
    st.download_button("↓ Export flagged readings", flags[show_cols].to_csv(index=False).encode("utf-8"), "flagged_sensor_readings.csv", "text/csv")

st.divider()
st.caption("NEXUS SENSOR INTELLIGENCE · COLLEGE PROJECT | Anomaly flags indicate unusual patterns, not confirmed sensor failure.")
