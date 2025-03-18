import requests
from bs4 import BeautifulSoup
import html2text


def browse_web(url):
    print("\n\n" + "#" * 40)
    print(f"Browsing web: {url}")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(response)
        return "Error retrieving website."
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove unnecessary elements
    for tag in soup(["script", "style", "nav", "footer", "iframe", ".ads"]):
        tag.decompose()
    
    # Extract title and main content
    title = soup.title.string.strip() if soup.title else ""
    main_content = (
        soup.select_one("article, main, .content, #content, .post") or soup.body
    )
    
    content = html2text.html2text(str(main_content)) if main_content else ""
    
    result = f"---\ntitle: '{title}'\n---\n\n{content}"
    print(result[:500])  # Preview first 500 characters
    return result


# Example usage
if __name__ == "__main__":
    url = "https://example.com"
    print(browse_web(url))
