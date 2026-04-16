from textblob import TextBlob

def preprocess(df, news_texts):

    # Base sentiment
    df["sentiment"] = df["text"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    # Handle external news safely
    if news_texts and len(news_texts) > 0:

        extra_sent = [
            TextBlob(str(t)).sentiment.polarity
            for t in news_texts if t
        ]

        if len(extra_sent) > 0:
            avg_extra = sum(extra_sent) / len(extra_sent)
        else:
            avg_extra = 0

    else:
        avg_extra = 0

    # Combine sentiment
    df["sentiment"] = df["sentiment"] + avg_extra

    # Convert to risk factor
    df["sentiment_impact"] = -df["sentiment"] * 10

    # Risk score
    df["risk_score"] = (
        0.6 * df["event_intensity"] +
        0.4 * df["sentiment_impact"]
    )

    # Labels
    def label(score):
        if score > 25:
            return 2
        elif score > 12:
            return 1
        else:
            return 0

    df["risk"] = df["risk_score"].apply(label)

    return df