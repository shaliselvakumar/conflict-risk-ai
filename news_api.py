import requests

API_KEY = "YOUR_API_KEY"

def fetch_news():

    try:
        url = f"https://newsapi.org/v2/everything?q=war OR conflict&apiKey={API_KEY}"
        res = requests.get(url).json()

        articles = res.get("articles", [])
        texts = [a["title"] for a in articles if a["title"]]

        return texts[:10]

    except:
        return ["conflict rising", "war escalation", "crisis situation"]