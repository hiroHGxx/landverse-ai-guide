from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set


def extract_links(html_content: str, base_url: str) -> List[str]:
    """
    Extract all unique absolute URLs from HTML content.
    
    Args:
        html_content: The HTML content as a string
        base_url: The base URL for converting relative URLs to absolute URLs
        
    Returns:
        A list of unique absolute URLs found in the HTML content
    """
    if not html_content or not base_url:
        return []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links: Set[str] = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            if not href:
                continue
                
            absolute_url = urljoin(base_url, href)
            
            parsed_url = urlparse(absolute_url)
            if parsed_url.scheme and parsed_url.netloc:
                links.add(absolute_url)
        
        return list(links)
    
    except Exception as e:
        print(f"Error parsing HTML content: {e}")
        return []