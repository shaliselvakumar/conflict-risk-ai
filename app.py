import streamlit as st
import numpy as np
import folium
import matplotlib.pyplot as plt
from streamlit.components.v1 import html

from data import load_data
from preprocess import preprocess
from train import train_model
from utils import get_label
from news_api import fetch_news

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Humanitarian AI",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CLEAN WHITE CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #1f2937;
}

/* Headings */
h1 {
    color: #0f172a;
}
h2, h3 {
    color: #2563eb;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #f9fafb;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
# 🌍 Humanitarian Risk Intelligence System  
### AI-powered platform for proactive conflict monitoring & decision support
""")

st.markdown("---")

# ---------------- LOAD DATA ----------------
news = fetch_news()
df = load_data()
df = preprocess(df, news)

# ---------------- MODEL ----------------
X = df[["event_intensity","sentiment_impact"]].values
y = df["risk_score"].values

model = train_model(X, y)

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("🌍 Countries", len(df))
col2.metric("🔴 High Risk Zones", int((df["risk"] == 2).sum()))
col3.metric("📊 Avg Risk", round(df["risk_score"].mean(), 2))

st.markdown("---")

# ---------------- INFO ----------------
st.markdown("""
### ⚙️ System Overview
This AI system analyzes conflict intensity and real-time sentiment signals to generate dynamic humanitarian risk scores across regions.
""")

st.markdown("---")

# ---------------- NEWS ----------------
st.subheader("📰 Live Conflict Signals")

if news:
    for n in news[:5]:
        st.write("•", n)
else:
    st.write("No live news available")

# ---------------- DASHBOARD ----------------
col1, col2 = st.columns(2)

# 📊 CHART
with col1:
    st.subheader("📊 Risk Distribution")

    counts = df["risk"].value_counts().reindex([0,1,2], fill_value=0)

    plt.figure()
    plt.bar(["Low","Medium","High"], counts.values)
    st.pyplot(plt)

# 🗺️ MAP
with col2:
    st.subheader("🗺️ Global Risk Map")

    coords = {
        "Ukraine":[48,31],"Gaza":[31.5,34.4],"Sudan":[15,30],
        "Syria":[35,38],"Yemen":[15,48],"Afghanistan":[33,65],
        "Iran":[32,53],"Israel":[31,35],"Pakistan":[30,70],
        "Ethiopia":[9,40],"Myanmar":[21,96],"Nigeria":[9,8],
        "Mali":[17,-4],"Somalia":[5,46],"DR Congo":[-2,23],
        "India":[20,78],"China":[35,103],"Russia":[60,100],
        "USA":[37,-95],"UK":[55,-3],"France":[46,2],
        "Germany":[51,10],"UAE":[24,54],
        "Saudi Arabia":[24,45],"Turkey":[39,35]
    }

    m = folium.Map(location=[20,0], zoom_start=2)

    for _, row in df.iterrows():
        if row["location"] in coords:
            color = ["green","orange","red"][row["risk"]]

            folium.Circle(
                location=coords[row["location"]],
                radius=200000,
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

    html(m._repr_html_(), height=450)

st.markdown("---")

# ---------------- DATA ----------------
st.subheader("📋 Processed Data")
st.dataframe(df)

st.markdown("---")

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predict Risk")

event = st.slider("Event Intensity", 0, 50, 10)
sent = st.slider("Sentiment", -1.0, 1.0, 0.0)

if st.button("Predict"):

    sent_imp = -sent * 10

    pred = model.predict(np.array([[event, sent_imp]]))[0]

    st.markdown("### AI Prediction Result")
    st.success(f"Risk Score: {round(pred,2)}")

    if pred > 25:
        st.error("🔴 HIGH RISK → Immediate action required")
    elif pred > 12:
        st.warning("🟡 MEDIUM RISK → Monitor closely")
    else:
        st.info("🟢 LOW RISK → Stable")

st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("Built with AI for humanitarian impact 🌍")