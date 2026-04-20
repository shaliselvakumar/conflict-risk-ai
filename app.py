import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from folium.plugins import HeatMap
import pycountry
import time
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

st.markdown("""
<style>
h1 {
    font-weight: 700;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AUTO REFRESH ----------------
st.experimental_rerun if st.button("🔄 Refresh Data") else None

# ---------------- DATA ----------------
countries = [c.name for c in pycountry.countries]

data = []
for c in countries:
    lat = np.random.uniform(-55, 70)
    lon = np.random.uniform(-180, 180)

    event = np.random.randint(1, 50)
    sentiment = np.random.randint(-5, 5)

    risk_score = event + (-sentiment * 4)

    if risk_score > 55:
        risk = 2
    elif risk_score > 30:
        risk = 1
    else:
        risk = 0

    data.append([c, lat, lon, event, sentiment, risk_score, risk])

df = pd.DataFrame(data, columns=[
    "location","lat","lon",
    "event_intensity","sentiment_impact",
    "risk_score","risk"
])

# ---------------- MODEL ----------------
X = df[["event_intensity","sentiment_impact"]]
y = df["risk_score"]

model = LinearRegression()
model.fit(X,y)

# ---------------- HEADER ----------------
st.title("🌍 AI Humanitarian Risk Intelligence System")
st.caption("Real-time global monitoring with AI + predictive analytics")

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🌍 Countries", len(df))
col2.metric("🚨 High Risk", int((df["risk"]==2).sum()))
col3.metric("⚠ Medium Risk", int((df["risk"]==1).sum()))
col4.metric("📊 Avg Risk", round(df["risk_score"].mean(),2))

st.markdown("---")

# ---------------- UAE PRIORITY ----------------
uae = df[df["location"]=="United Arab Emirates"]

if not uae.empty:
    uae_risk = float(uae["risk_score"].values[0])

    st.subheader("🇦🇪 UAE Intelligence Status")

    if uae_risk > 55:
        st.error(f"HIGH RISK → {round(uae_risk,2)}")
    elif uae_risk > 30:
        st.warning(f"MEDIUM RISK → {round(uae_risk,2)}")
    else:
        st.success(f"LOW RISK → {round(uae_risk,2)}")

st.markdown("---")

# ---------------- ALERT CENTER ----------------
st.subheader("🚨 AI Alert Center")

high = df[df["risk"]==2].sort_values("risk_score", ascending=False).head(10)

for _,r in high.iterrows():
    st.error(f"{r['location']} → Risk {round(r['risk_score'],2)}")

st.markdown("---")
import requests

st.markdown("---")
st.subheader("📰 Live Conflict Intelligence Feed")

def get_news():
    try:
        url = "https://newsapi.org/v2/everything?q=conflict OR war OR crisis&sortBy=publishedAt&apiKey=e530a68fd1a4477d947fa57ec6d2f981"
        res = requests.get(url).json()
        return res["articles"][:5]
    except:
        return []

news = get_news()

if news:
    for article in news:
        st.markdown(f"""
        🔴 **{article['title']}**  
        {article['source']['name']}  
        """)
else:
    st.info("No live news available")
# ---------------- INSANE MAP ----------------
st.subheader("🗺️ Global Intelligence Map")

colormap = cm.LinearColormap(
    ["green","yellow","red"],
    vmin=df["risk_score"].min(),
    vmax=df["risk_score"].max()
)

m = folium.Map(location=[20,0], zoom_start=2, tiles="cartodb dark_matter")

for _, row in df.iterrows():
    color = colormap(row["risk_score"])

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6 + row["risk"]*2,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=f"{row['location']} | Risk: {round(row['risk_score'],2)}"
    ).add_to(m)

heat_data = [[row["lat"], row["lon"], row["risk_score"]] for _, row in df.iterrows()]

HeatMap(heat_data, radius=15, blur=10).add_to(m)

colormap.add_to(m)

st_folium(m, width=1200, height=550)

st.markdown("---")

# ---------------- ANALYTICS ----------------
st.subheader("📊 Risk Analytics")

fig = px.scatter(
    df,
    x="event_intensity",
    y="risk_score",
    color="risk",
    hover_name="location"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------- PREDICTION ----------------
st.subheader("🔮 AI Prediction Tool")

col1, col2 = st.columns(2)

event = col1.slider("Event Intensity",0,50,10)
sentiment = col2.slider("Sentiment",-10,10,0)

if st.button("Run AI Prediction"):

    with st.spinner("AI analyzing..."):
        time.sleep(1.5)

    pred = float(model.predict(np.array([[event, sentiment]]))[0])

    st.subheader("🧠 AI Explanation")

    st.write(f"Event Impact Contribution: {event}")
    st.write(f"Sentiment Impact Contribution: {-sentiment*4}")

    if pred > 55:
        st.error(f"HIGH RISK → {round(pred,2)}")
    elif pred > 30:
        st.warning(f"MEDIUM RISK → {round(pred,2)}")
    else:
        st.success(f"LOW RISK → {round(pred,2)}")

st.markdown("---")
# ---------------- EVACUATION ROUTE ----------------
st.markdown("---")
st.subheader("🧭 AI-Assisted Evacuation Planning")

origin = st.text_input("Enter Risk Zone (City/Country)")
destination = st.text_input("Enter Safe Zone")

if st.button("Generate Route"):
    if origin and destination:
        map_url = f"https://www.google.com/maps/dir/{origin}/{destination}"
        
        st.success("Optimal evacuation route generated")
        st.markdown(f"[Open Route in Google Maps]({map_url})")
    else:
        st.warning("Please enter both locations")
# ---------------- DATA ----------------
with st.expander("📁 View Full Dataset"):
    st.dataframe(df)