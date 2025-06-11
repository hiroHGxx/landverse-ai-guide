import requests
from typing import Optional


def fetch_html(url: str) -> Optional[str]:
    """
    Fetch HTML content from the given URL.
    
    Args:
        url: The URL to fetch HTML content from
        
    Returns:
        The HTML content as a string, or None if the request fails
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML from {url}: {e}")
        return None