import requests

# Web search function using DuckDuckGo Instant Answer API
def search_query(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url).json()
    return response.get("RelatedTopics", [{}])[0].get("Text", "No results found")

# (Optional) You can implement local QA later using Ollama if needed
