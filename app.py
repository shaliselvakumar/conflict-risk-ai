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

st.set_page_config(page_title="Humanitarian AI", layout="wide")

st.title("🌍 Proactive Humanitarian Risk Intelligence System")
st.markdown("AI system integrating conflict data, news signals, and geospatial intelligence.")

# ---------------------------
# LOAD DATA
# ---------------------------
news = fetch_news()
df = load_data()
df = preprocess(df, news)

# ---------------------------
# TRAIN MODEL
# ---------------------------
X = df[["event_intensity","sentiment_impact"]].values
y = df["risk_score"].values

model = train_model(X, y)

# ---------------------------
# NEWS PANEL
# ---------------------------
st.subheader("📰 Live Conflict Signals")

if news:
    for n in news[:5]:
        st.write("•", n)
else:
    st.write("No live news available (using fallback data)")

# ---------------------------
# DASHBOARD
# ---------------------------
col1, col2 = st.columns(2)

# 📊 CHART (FIXED)
with col1:
    st.subheader("📊 Risk Distribution")

    # FIX: ensure all 3 categories exist
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

# ---------------------------
# DATA TABLE
# ---------------------------
st.subheader("📋 Processed Data")
st.dataframe(df)

# ---------------------------
# PREDICTION
# ---------------------------
st.subheader("🔮 Predict Risk")

event = st.slider("Event Intensity", 0, 50, 10)
sent = st.slider("Sentiment", -1.0, 1.0, 0.0)

if st.button("Predict"):

    sent_imp = -sent * 10

    pred = model.predict(np.array([[event, sent_imp]]))[0][0]

    st.subheader(get_label(pred))
    st.write(f"Risk Score: {round(pred,2)}")

    if pred > 25:
        st.error("⚠️ High Risk → Immediate humanitarian response needed")
    elif pred > 12:
        st.warning("⚠️ Medium Risk → Monitor closely")
    else:
        st.success("✅ Low Risk → Stable")