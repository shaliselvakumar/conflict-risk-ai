import streamlit as st
import numpy as np
import plotly.express as px
import folium
from streamlit.components.v1 import html
import time
from sklearn.linear_model import LinearRegression
from data_loader import load_data

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- GLASS + GRID UI ----------------
st.markdown("""
<style>

/* -------- BACKGROUND -------- */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    color: #0f172a;
    overflow-x: hidden;
}

/* -------- ANIMATED BLOBS -------- */
.stApp::after {
    content: "";
    position: fixed;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.3), transparent);
    top: -100px;
    left: -100px;
    filter: blur(120px);
    animation: moveBlob 10s infinite alternate;
    z-index: -1;
}

@keyframes moveBlob {
    from { transform: translate(0,0); }
    to { transform: translate(200px,200px); }
}

/* -------- GRID -------- */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background-image:
        linear-gradient(rgba(0,0,0,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: -1;
}

/* -------- GLASS CARD -------- */
.card {
    background: rgba(255,255,255,0.6);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(99,102,241,0.2);
}

/* -------- BUTTON -------- */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    border-radius: 12px;
    font-weight: bold;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(99,102,241,0.4);
}

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
}

/* -------- HEADINGS -------- */
h1, h2, h3 {
    color: #0f172a;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = load_data()

# ---------------- MODEL ----------------
X = df[["event_intensity","sentiment_impact"]]
y = df["risk_score"]

model = LinearRegression()
model.fit(X,y)

# ---------------- NAV ----------------
menu = st.sidebar.radio("Navigation", ["Home","Dashboard","Prediction","Data"])

# ---------------- HOME ----------------
st.title("🌍 Humanitarian AI Intelligence")

st.markdown("""
<div class="card">
<h2>🚀 AI-Powered Conflict Monitoring</h2>
<p>Predicting humanitarian risks before escalation using real-time global data and AI.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card">
<h3>🌍 Countries</h3>
<h2>{len(df)}</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
<h3>🚨 High Risk</h3>
<h2>{int((df["risk"]==2).sum())}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
<h3>📊 Avg Risk</h3>
<h2>{round(df["risk_score"].mean(),2)}</h2>
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":

    st.title("📊 Global Risk Dashboard")

    st.subheader("🚨 Alerts")

    for _,row in df[df["risk"]==2].iterrows():
        st.error(f"{row['location']} HIGH RISK")

    st.markdown("---")

    st.subheader("📈 Risk Trends")

    fig = px.area(
    df,
    x="location",
    y="risk_score",
    title="Global Risk Distribution",
)
st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("🗺️ Risk Map")

    m = folium.Map(location=[20,0], zoom_start=2)

    for _,row in df.iterrows():
        lat = np.random.uniform(-60,60)
        lon = np.random.uniform(-180,180)

        color = ["green","orange","red"][row["risk"]]

        folium.CircleMarker(
            location=[lat,lon],
            radius=6,
            color=color,
            fill=True
        ).add_to(m)

    html(m._repr_html_(), height=500)

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    st.title("🔮 Risk Prediction")

    event = st.slider("Event Intensity",0,50,10)
    sentiment = st.slider("Sentiment",-10,10,0)

    if st.button("Analyze"):

    with st.spinner("🤖 AI analyzing global patterns..."):
        time.sleep(2)

    pred = float(model.predict(np.array([[event, sentiment]]))[0])

       st.subheader("🚨 AI Alert Center")

high = df[df["risk"] == 2]
medium = df[df["risk"] == 1]

# -------- HIGH RISK --------
if len(high) > 0:
    st.markdown("### 🔴 High Risk Zones")

    for _, row in high.iterrows():
        st.markdown(f"""
        <div style="
        padding:15px;
        border-radius:12px;
        background: rgba(239,68,68,0.1);
        border-left: 6px solid #ef4444;
        margin-bottom:10px;">
        <b>{row['location']}</b><br>
        Risk Score: {round(row['risk_score'],2)}<br>
        Status: Immediate attention required
        </div>
        """, unsafe_allow_html=True)

# -------- MEDIUM RISK --------
if len(medium) > 0:
    st.markdown("### 🟡 Medium Risk Zones")

    for _, row in medium.iterrows():
        st.markdown(f"""
        <div style="
        padding:15px;
        border-radius:12px;
        background: rgba(234,179,8,0.1);
        border-left: 6px solid #eab308;
        margin-bottom:10px;">
        <b>{row['location']}</b><br>
        Risk Score: {round(row['risk_score'],2)}<br>
        Status: Monitoring required
        </div>
        """, unsafe_allow_html=True)

# -------- NO ALERT --------
if len(high) == 0 and len(medium) == 0:
    st.success("🟢 No active high-risk alerts")
# ---------------- DATA ----------------
else:
    st.dataframe(df)