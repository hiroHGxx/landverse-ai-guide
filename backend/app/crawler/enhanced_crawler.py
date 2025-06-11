"""
Enhanced web crawler with async support, better error handling, and performance optimizations
"""
import asyncio
import aiohttp
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import List, Set, Tuple, Optional, AsyncGenerator
from dataclasses import dataclass

from ..utils import get_logger, LoggerMixin, handle_exceptions
from ..config import get_config
from .parser import extract_links


@dataclass
class CrawlResult:
    """Result of a single page crawl"""
    url: str
    content: str
    status_code: int
    error: Optional[str] = None


class EnhancedWebCrawler(LoggerMixin):
    """Enhanced web crawler with async support and better error handling"""
    
    def __init__(self, config=None):
        self.config = config or get_config().crawler
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _create_session(self):
        """Create aiohttp session with appropriate settings"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            limit_per_host=self.config.max_concurrent // 2
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'RAG-Crawler/1.0 (+https://github.com/your-repo)'
            }
        )
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    @handle_exceptions()
    async def fetch_single_url(self, url: str) -> CrawlResult:
        """Fetch a single URL with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                
                async with self.session.get(url) as response:
                    content = await response.text()
                    
                    if response.status == 200:
                        self.logger.debug(f"Successfully fetched {url}")
                        return CrawlResult(
                            url=url,
                            content=content,
                            status_code=response.status
                        )
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout for {url} on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        error_msg = f"Failed to fetch after {self.config.max_retries} attempts"
        self.logger.error(f"{error_msg}: {url}")
        self.failed_urls.add(url)
        
        return CrawlResult(
            url=url,
            content="",
            status_code=0,
            error=error_msg
        )
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL belongs to allowed domain"""
        try:
            parsed_url = urlparse(url)
            domain = self.config.domain
            return (parsed_url.netloc == domain or 
                   parsed_url.netloc.endswith(f'.{domain}'))
        except Exception as e:
            self.logger.warning(f"Error parsing URL {url}: {e}")
            return False
    
    async def _crawl_level(self, urls: List[str], semaphore: asyncio.Semaphore) -> List[CrawlResult]:
        """Crawl a single level of URLs concurrently"""
        async def fetch_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                # Rate limiting
                await asyncio.sleep(self.config.delay_between_requests)
                return await self.fetch_single_url(url)
        
        # Filter out already visited URLs
        new_urls = [url for url in urls if url not in self.visited_urls]
        self.visited_urls.update(new_urls)
        
        if not new_urls:
            return []
        
        self.logger.info(f"Crawling {len(new_urls)} URLs concurrently")
        
        # Execute all fetches concurrently
        tasks = [fetch_with_semaphore(url) for url in new_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        crawl_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
            elif isinstance(result, CrawlResult):
                crawl_results.append(result)
        
        return crawl_results
    
    @handle_exceptions()
    async def crawl_website_async(self, start_url: str = None, max_depth: int = None, 
                                 domain: str = None) -> AsyncGenerator[CrawlResult, None]:
        """
        Asynchronously crawl website with specified depth and domain restrictions.
        Yields results as they become available.
        """
        # Use config defaults if not provided
        start_url = start_url or self.config.start_url
        max_depth = max_depth if max_depth is not None else self.config.max_depth
        domain = domain or self.config.domain
        
        if max_depth < 0:
            return
        
        self.logger.info(f"Starting async crawl of {start_url} with max_depth={max_depth}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Initialize queue with starting URL
        current_level_urls = [start_url]
        
        for depth in range(max_depth + 1):
            if not current_level_urls:
                break
            
            self.logger.info(f"Crawling depth {depth} with {len(current_level_urls)} URLs")
            
            # Crawl current level
            results = await self._crawl_level(current_level_urls, semaphore)
            
            # Yield successful results
            next_level_urls = []
            for result in results:
                if result.error is None and result.content:
                    yield result
                    
                    # Extract links for next level (if not at max depth)
                    if depth < max_depth:
                        try:
                            links = extract_links(result.content, result.url)
                            valid_links = [
                                link for link in links 
                                if (self._is_allowed_domain(link) and 
                                   link not in self.visited_urls and
                                   link not in self.failed_urls)
                            ]
                            next_level_urls.extend(valid_links)
                        except Exception as e:
                            self.logger.warning(f"Error extracting links from {result.url}: {e}")
            
            # Prepare for next level
            current_level_urls = list(set(next_level_urls))  # Remove duplicates
        
        self.logger.info(f"Crawling completed. Visited: {len(self.visited_urls)}, Failed: {len(self.failed_urls)}")
    
    @handle_exceptions()
    async def crawl_website(self, start_url: str = None, max_depth: int = None, 
                           domain: str = None) -> List[Tuple[str, str]]:
        """
        Crawl website and return list of (url, content) tuples.
        This is the main public method that maintains backward compatibility.
        """
        results = []
        
        async for result in self.crawl_website_async(start_url, max_depth, domain):
            if result.error is None and result.content:
                results.append((result.url, result.content))
        
        return results


# Backward compatibility function
@handle_exceptions()
async def crawl_website_async(start_url: str, max_depth: int, domain: str) -> List[Tuple[str, str]]:
    """Async version of crawl_website function for backward compatibility"""
    async with EnhancedWebCrawler() as crawler:
        return await crawler.crawl_website(start_url, max_depth, domain)


# Synchronous wrapper for backward compatibility
@handle_exceptions()
def crawl_website(start_url: str, max_depth: int, domain: str) -> List[Tuple[str, str]]:
    """Synchronous crawl_website function for backward compatibility"""
    return asyncio.run(crawl_website_async(start_url, max_depth, domain))