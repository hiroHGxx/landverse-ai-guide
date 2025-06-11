from collections import deque
from urllib.parse import urlparse
from typing import List, Set, Tuple, Dict
from bs4 import BeautifulSoup

from .fetcher import fetch_html
from .parser import extract_links


def crawl_website(start_url: str, max_depth: int, domain: str) -> List[Tuple[str, str]]:
    """
    Crawl a website starting from the given URL with specified depth and domain restrictions.
    
    Args:
        start_url: The starting URL to begin crawling
        max_depth: Maximum depth to crawl (0 = start page only, 1 = start + linked pages)
        domain: The domain to restrict crawling to
        
    Returns:
        List of tuples containing (URL, text_content) for each successfully crawled page
    """
    if max_depth < 0:
        return []
    
    visited_urls: Set[str] = set()
    results: List[Tuple[str, str]] = []
    
    # Queue stores tuples of (url, current_depth)
    url_queue: deque = deque([(start_url, 0)])
    
    while url_queue:
        current_url, current_depth = url_queue.popleft()
        
        # Skip if already visited
        if current_url in visited_urls:
            continue
            
        # Skip if URL is not from the allowed domain
        if not _is_allowed_domain(current_url, domain):
            continue
            
        # Mark as visited
        visited_urls.add(current_url)
        
        # Fetch HTML content
        html_content = fetch_html(current_url)
        if html_content is None:
            continue
            
        # Extract text content
        text_content = _extract_text_content(html_content)
        results.append((current_url, text_content))
        
        # If we haven't reached max depth, extract links for next level
        if current_depth < max_depth:
            links = extract_links(html_content, current_url)
            for link in links:
                if link not in visited_urls and _is_allowed_domain(link, domain):
                    url_queue.append((link, current_depth + 1))
    
    return results


def _is_allowed_domain(url: str, allowed_domain: str) -> bool:
    """
    Check if the URL belongs to the allowed domain.
    
    Args:
        url: The URL to check
        allowed_domain: The allowed domain
        
    Returns:
        True if the URL is from the allowed domain, False otherwise
    """
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc == allowed_domain or parsed_url.netloc.endswith(f'.{allowed_domain}')
    except Exception:
        return False


def _extract_text_content(html_content: str) -> str:
    """
    Extract clean text content from HTML, removing all tags.
    
    Args:
        html_content: The HTML content as a string
        
    Returns:
        Clean text content with HTML tags removed
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean up whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error extracting text content: {e}")
        return ""