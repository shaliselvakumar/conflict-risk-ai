import pandas as pd
import requests
from textblob import TextBlob

# ---------------- GDELT DATA ----------------
def load_gdelt():
    url = "http://data.gdeltproject.org/events/20240315.export.CSV.zip"

    try:
        df = pd.read_csv(url, sep='\t', header=None, low_memory=False)

        df = df[[1, 7]]  # Actor + EventCode
        df.columns = ["location", "event_code"]

        df = df.dropna().head(200)

        return df

    except:
        # fallback if API fails
        return pd.DataFrame({
            "location":["Ukraine","Gaza","Sudan","India","USA"],
            "event_code":[40,45,35,10,8]
        })


# ---------------- NEWS DATA ----------------
def fetch_news():
    API_KEY = "e530a68fd1a4477d947fa57ec6d2f981"

    url = f"https://newsapi.org/v2/everything?q=war OR conflict&apiKey={API_KEY}"

    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])

        headlines = [a["title"] for a in articles[:20] if a["title"]]

        return headlines

    except:
        return []


# ---------------- SENTIMENT ----------------
def get_sentiment(headlines):
    scores = []

    for h in headlines:
        blob = TextBlob(h)
        scores.append(blob.sentiment.polarity)

    if len(scores) == 0:
        return 0

    return sum(scores)/len(scores)


# ---------------- FINAL DATA ----------------
def build_dataset():
    df = load_gdelt()
    news = fetch_news()

    sentiment = get_sentiment(news)

    df["event_intensity"] = df["event_code"] % 50
    df["sentiment_impact"] = sentiment * 10

    df["risk_score"] = df["event_intensity"] + (-df["sentiment_impact"]*2)

    def label(x):
        if x > 50: return 2
        elif x > 25: return 1
        return 0

    df["risk"] = df["risk_score"].apply(label)

    return df