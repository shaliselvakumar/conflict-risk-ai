import streamlit as st
import numpy as np
import plotly.express as px
import folium
import requests
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

from data import load_data
from preprocess import preprocess
from train import train_model
from news_api import fetch_news

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Humanitarian AI", layout="wide")

# ---------------- LOAD ANIMATION ----------------
def load_lottie(url):
    return requests.get(url).json()

lottie_ai = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        "🌍 Humanitarian AI",
        ["Home", "Dashboard", "Prediction", "Data"],
        icons=["house", "bar-chart", "cpu", "table"],
        default_index=0
    )

# ---------------- LOAD DATA ----------------
news = fetch_news()
df = load_data()
df = preprocess(df, news)

X = df[["event_intensity","sentiment_impact"]].values
y = df["risk_score"].values
model = train_model(X, y)

# ---------------- MAP ----------------
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

# ---------------- HOME ----------------
if selected == "Home":

    col1, col2 = st.columns([2,1])

    with col1:
        st.title("🌍 Humanitarian Risk Intelligence System")

        st.markdown("""
        ### AI-powered platform for proactive crisis detection
        
        Predict risks before they escalate using AI-driven insights.
        """)

        st.markdown("---")

        st.markdown("""
        ### 🚨 Why This Matters
        Early prediction saves lives by enabling faster response.
        
        ### ⚙️ How It Works
        - Collects global conflict data  
        - Analyzes sentiment signals  
        - Generates AI risk scores  
        - Visualizes global threats  
        """)

    with col2:
        st_lottie(lottie_ai, height=250)

    st.success("System Status: ✅ Active")

# ---------------- DASHBOARD ----------------
elif selected == "Dashboard":

    st.title("📊 Global Risk Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries", len(df))
    col2.metric("High Risk", int((df["risk"] == 2).sum()))
    col3.metric("Avg Risk", round(df["risk_score"].mean(), 2))

    st.markdown("---")

    st.subheader("📰 Live Conflict Signals")
    if news:
        for n in news[:5]:
            st.write("•", n)

    st.markdown("---")

    st.subheader("📊 Risk Distribution")

    counts = df["risk"].value_counts().reindex([0,1,2], fill_value=0)

    fig = px.bar(
        x=["Low","Medium","High"],
        y=counts.values,
        color=["Low","Medium","High"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🗺️ Global Risk Map")
    html(m._repr_html_(), height=500)

# ---------------- PREDICTION ----------------
elif selected == "Prediction":

    st.title("🔮 Risk Prediction Engine")

    event = st.slider("Event Intensity", 0, 50, 10)
    sent = st.slider("Sentiment", -1.0, 1.0, 0.0)

    if st.button("Run AI Prediction"):

        sent_imp = -sent * 10
        pred = model.predict(np.array([[event, sent_imp]]))[0]

        st.subheader(f"Risk Score: {round(pred,2)}")

        if pred > 25:
            st.error("🔴 HIGH RISK → Immediate action required")
        elif pred > 12:
            st.warning("🟡 MEDIUM RISK → Monitor closely")
        else:
            st.success("🟢 LOW RISK → Stable")

# ---------------- DATA ----------------
elif selected == "Data":

    st.title("📋 Data Explorer")
    st.dataframe(df)