import requests
import json
from pydantic import BaseModel


# Define a Pydantic model for input validation
class SearchGoogleSchema(BaseModel):
    query: str


# Define the searchGoogle function
def search_google(query: str):
    # Initialize the API endpoint and API key
    api_key = "YOUR_SERP_API_KEY"  # Replace with your actual API key
    url = "https://serpapi.com/search"
    
    # Make the request to SerpAPI
    response = requests.get(url, params={
        "engine": "google",
        "api_key": api_key,
        "q": query
    })
    
    # Parse the JSON response
    resp = response.json()

    # Get the top 5 organic search results
    results = [{"title": el['title'], "url": el['link']} for el in resp.get('organic_results', [])[:5]]
    
    # Convert results to string format
    string_result = json.dumps(results)
    print(string_result)
    return string_result


# Example usage
search_google("Python tutorial")
