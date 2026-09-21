import os
import re
import requests
from bs4 import BeautifulSoup

def fetch_job_details(input_text: str) -> str:
    """Detects if the input contains a URL and scrapes it; otherwise returns the raw text."""
    url_match = re.search(r"https?://[^\s]+", input_text)
    if not url_match:
        return input_text

    url = url_match.group(0)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = " ".join(soup.stripped_strings)
        return text[:4000]
    except Exception as e:
        return f"Could not scrape URL ({url}). Using raw input. Error: {str(e)}"

def load_user_profile(file_path: str = "profile.md") -> str:
    """Reads your baseline profile for grounding."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No profile document found."

def save_document_artifact(thread_id: str, content: str) -> str:
    """Saves the finalized document to the local outputs directory."""
    os.makedirs("outputs", exist_ok=True)
    file_path = os.path.join("outputs", f"{thread_id}_cover_letter.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path