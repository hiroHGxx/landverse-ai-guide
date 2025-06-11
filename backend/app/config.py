"""
Configuration management for RAG system
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent.parent / '.env')


@dataclass
class CrawlerConfig:
    """Configuration for web crawler"""
    start_url: str = "https://guide.rolg.maxion.gg"
    max_depth: int = 2
    domain: str = "guide.rolg.maxion.gg"
    max_concurrent: int = 10
    timeout: int = 10
    max_retries: int = 3
    delay_between_requests: float = 1.0


@dataclass
class ChunkerConfig:
    """Configuration for text chunker"""
    max_chunk_size: int = 1000
    overlap_size: int = 100
    language: str = "mixed"  # "ja", "en", "mixed"


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    model_name: str = "all-MiniLM-L6-v2"
    database_path: str = "data/vector_db"
    use_gpu: bool = False
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    cache_size: int = 1000


@dataclass
class LLMConfig:
    """Configuration for LLM (Gemini)"""
    model_name: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    translation_cache_size: int = 1000


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "rag_system.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class RAGConfig:
    """Main configuration class"""
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set API key from environment if not provided
        if self.llm.api_key is None:
            self.llm.api_key = os.getenv("GEMINI_API_KEY")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'RAGConfig':
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Create default config file
            default_config = cls()
            default_config.save_to_file(config_path)
            return default_config
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            return cls()
        
        # Create configuration objects
        crawler_config = CrawlerConfig(**config_data.get('crawler', {}))
        chunker_config = ChunkerConfig(**config_data.get('chunker', {}))
        vector_db_config = VectorDBConfig(**config_data.get('vector_db', {}))
        llm_config = LLMConfig(**config_data.get('llm', {}))
        logging_config = LoggingConfig(**config_data.get('logging', {}))
        
        return cls(
            crawler=crawler_config,
            chunker=chunker_config,
            vector_db=vector_db_config,
            llm=llm_config,
            logging=logging_config
        )
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_data = {
            'crawler': {
                'start_url': self.crawler.start_url,
                'max_depth': self.crawler.max_depth,
                'domain': self.crawler.domain,
                'max_concurrent': self.crawler.max_concurrent,
                'timeout': self.crawler.timeout,
                'max_retries': self.crawler.max_retries,
                'delay_between_requests': self.crawler.delay_between_requests,
            },
            'chunker': {
                'max_chunk_size': self.chunker.max_chunk_size,
                'overlap_size': self.chunker.overlap_size,
                'language': self.chunker.language,
            },
            'vector_db': {
                'model_name': self.vector_db.model_name,
                'database_path': self.vector_db.database_path,
                'use_gpu': self.vector_db.use_gpu,
                'index_type': self.vector_db.index_type,
                'cache_size': self.vector_db.cache_size,
            },
            'llm': {
                'model_name': self.llm.model_name,
                'max_tokens': self.llm.max_tokens,
                'temperature': self.llm.temperature,
                'translation_cache_size': self.llm.translation_cache_size,
                # Don't save API key to file for security
            },
            'logging': {
                'level': self.logging.level,
                'log_dir': self.logging.log_dir,
                'log_file': self.logging.log_file,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count,
                'format': self.logging.format,
            }
        }
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        # LLM Configuration
        if os.getenv("GEMINI_API_KEY"):
            self.llm.api_key = os.getenv("GEMINI_API_KEY")
        
        if os.getenv("LLM_MODEL_NAME"):
            self.llm.model_name = os.getenv("LLM_MODEL_NAME")
        
        # Database path
        if os.getenv("VECTOR_DB_PATH"):
            self.vector_db.database_path = os.getenv("VECTOR_DB_PATH")
        
        # Crawler configuration
        if os.getenv("CRAWLER_START_URL"):
            self.crawler.start_url = os.getenv("CRAWLER_START_URL")
        
        if os.getenv("CRAWLER_DOMAIN"):
            self.crawler.domain = os.getenv("CRAWLER_DOMAIN")
        
        # Logging level
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")


# Global configuration instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = RAGConfig.from_file("config/rag_config.yaml")
        _config.update_from_env()
    return _config


def reload_config(config_path: str = "config/rag_config.yaml") -> RAGConfig:
    """Reload configuration from file"""
    global _config
    _config = RAGConfig.from_file(config_path)
    _config.update_from_env()
    return _config