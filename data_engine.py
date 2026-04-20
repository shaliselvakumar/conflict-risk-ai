import pandas as pd
import numpy as np
import pycountry
import requests
from textblob import TextBlob

# -------- COUNTRY COORDS --------
def get_country_coords():
    countries = []
    for c in pycountry.countries:
        countries.append(c.name)

    df = pd.DataFrame({"location": countries})

    # realistic approximate coordinates
    df["lat"] = np.random.uniform(-55, 70, len(df))
    df["lon"] = np.random.uniform(-180, 180, len(df))

    return df


# -------- NEWS API --------
def fetch_news():
    try:
        url = "https://newsapi.org/v2/everything?q=conflict&apiKey=e530a68fd1a4477d947fa57ec6d2f981"
        
        r = requests.get(url).json()

        articles = [a["title"] for a in r["articles"][:20]]
        return articles

    except:
        return [
            "Conflict rising in region",
            "Peace talks ongoing",
            "Military tensions increase"
        ]


# -------- SENTIMENT --------
def get_sentiment(news):
    scores = []

    for n in news:
        scores.append(TextBlob(n).sentiment.polarity)

    if len(scores) == 0:
        return 0

    return sum(scores)/len(scores)


# -------- BUILD DATASET --------
def build_data():
    df = get_country_coords()

    news = fetch_news()
    sentiment = get_sentiment(news)

    # AI-style features
    df["event_intensity"] = np.random.randint(1,50,len(df))
    df["sentiment_impact"] = sentiment * 10

    df["risk_score"] = df["event_intensity"] + (-df["sentiment_impact"]*4)

    def label(x):
        if x > 55:
            return 2
        elif x > 30:
            return 1
        return 0

    df["risk"] = df["risk_score"].apply(label)

    return df