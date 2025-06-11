"""
Enhanced vector database with advanced indexing, caching, and performance optimizations
"""
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
import hashlib

from ..utils import get_logger, LoggerMixin, handle_exceptions
from ..config import get_config


class EnhancedVectorDatabase(LoggerMixin):
    """
    Enhanced vector database with advanced indexing and caching capabilities
    """
    
    def __init__(self, config=None):
        self.config = config or get_config().vector_db
        self.model_name = self.config.model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.text_chunks: List[str] = []
        self.dimension: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def _get_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model"""
        if self.model is None:
            self.logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def _create_cache_key(self, text: str) -> str:
        """Create cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _encode_cached(self, text: str) -> np.ndarray:
        """Cached text encoding"""
        return self._get_model().encode([text])[0]
    
    def _encode_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Encode batch of texts with caching"""
        if not use_cache:
            return self._get_model().encode(texts, show_progress_bar=True)
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._create_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Encode uncached texts
        if uncached_texts:
            self.logger.info(f"Encoding {len(uncached_texts)} uncached texts")
            new_embeddings = self._get_model().encode(uncached_texts, show_progress_bar=True)
            
            # Update cache and embeddings list
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cache_key = self._create_cache_key(texts[idx])
                self._embedding_cache[cache_key] = embedding
                embeddings[idx] = embedding
        
        return np.array(embeddings).astype('float32')
    
    def _create_advanced_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create advanced FAISS index based on configuration"""
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        if self.config.index_type == "flat":
            # Simple flat index (good for small datasets)
            index = faiss.IndexFlatIP(dimension)
            
        elif self.config.index_type == "ivf":
            # IVF index (good for medium to large datasets)
            n_clusters = min(100, max(10, n_vectors // 50))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            
            # Train the index
            self.logger.info(f"Training IVF index with {n_clusters} clusters")
            index.train(embeddings)
            
        elif self.config.index_type == "hnsw":
            # HNSW index (good for fast approximate search)
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
            
        else:
            self.logger.warning(f"Unknown index type {self.config.index_type}, using flat")
            index = faiss.IndexFlatIP(dimension)
        
        # GPU support
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.logger.info("Moving index to GPU")
            gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        
        return index
    
    @handle_exceptions()
    def build_database(self, text_chunks: List[str], save_path: str) -> None:
        """Build vector database with enhanced indexing"""
        if not text_chunks:
            raise ValueError("Text chunks list cannot be empty")
        
        self.logger.info(f"Building vector database with {len(text_chunks)} chunks")
        
        # Encode text chunks
        embeddings = self._encode_batch(text_chunks)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Get dimension
        self.dimension = embeddings.shape[1]
        
        # Create advanced index
        self.index = self._create_advanced_index(embeddings)
        
        # Add embeddings to index
        self.logger.info("Adding embeddings to index")
        self.index.add(embeddings)
        
        # Store text chunks and metadata
        self.text_chunks = text_chunks
        self.metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'index_type': self.config.index_type,
            'num_chunks': len(text_chunks),
            'use_gpu': self.config.use_gpu
        }
        
        # Save to files
        self._save_database(save_path)
        
        self.logger.info(f"Vector database built and saved to {save_path}")
    
    @handle_exceptions()
    def search_similar_chunks(self, query: str, database_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks with enhanced performance"""
        # Load database if not already loaded
        if self.index is None or not self.text_chunks:
            self._load_database(database_path)
        
        if self.index is None:
            raise ValueError(f"Could not load database from {database_path}")
        
        self.logger.debug(f"Searching for similar chunks: '{query}'")
        
        # Encode query
        query_embedding = self._encode_batch([query], use_cache=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search with appropriate k
        search_k = min(top_k, len(self.text_chunks))
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(score)))
        
        self.logger.debug(f"Found {len(results)} similar chunks")
        return results
    
    def _save_database(self, save_path: str) -> None:
        """Save database with metadata"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert GPU index to CPU for saving
        index_to_save = self.index
        if self.config.use_gpu and hasattr(self.index, 'index'):
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        
        # Save FAISS index
        faiss.write_index(index_to_save, f"{save_path}.faiss")
        
        # Save metadata and text chunks
        save_data = {
            'text_chunks': self.text_chunks,
            'metadata': self.metadata,
            'embedding_cache': dict(list(self._embedding_cache.items())[:self.config.cache_size])
        }
        
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Database saved to {save_path}")
    
    def _load_database(self, database_path: str) -> None:
        """Load database with metadata"""
        faiss_path = f"{database_path}.faiss"
        metadata_path = f"{database_path}.pkl"
        
        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Database files not found at {database_path}")
        
        self.logger.info(f"Loading database from {database_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        
        # Move to GPU if configured
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
        
        # Load metadata and text chunks
        with open(metadata_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.text_chunks = save_data['text_chunks']
        
        # Handle both old and new metadata formats
        if 'metadata' in save_data:
            self.metadata = save_data['metadata']
            self.dimension = self.metadata['dimension']
            self.model_name = self.metadata['model_name']
        else:
            # Legacy format compatibility
            self.metadata = {
                'dimension': save_data.get('dimension', 384),  # Default for all-MiniLM-L6-v2
                'model_name': save_data.get('model_name', self.config.model_name),
                'chunks_count': len(self.text_chunks)
            }
            self.dimension = self.metadata['dimension']
            self.model_name = self.metadata['model_name']
        
        # Load embedding cache if available
        if 'embedding_cache' in save_data:
            self._embedding_cache = save_data['embedding_cache']
        
        # Ensure model is loaded
        self._get_model()
        
        self.logger.info(f"Database loaded successfully. Chunks: {len(self.text_chunks)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'num_chunks': len(self.text_chunks),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'cache_size': len(self._embedding_cache),
            'index_loaded': self.index is not None,
        }
        
        if self.metadata:
            stats.update(self.metadata)
        
        return stats


# Convenience functions for backward compatibility
@handle_exceptions()
def build_vector_database(text_chunks: List[str], save_path: str, model_name: str = None) -> None:
    """Build vector database - backward compatibility function"""
    config = get_config().vector_db
    if model_name:
        config.model_name = model_name
    
    db = EnhancedVectorDatabase(config)
    db.build_database(text_chunks, save_path)


@handle_exceptions()
def search_vector_database(query: str, database_path: str, top_k: int = 5, model_name: str = None) -> List[Tuple[str, float]]:
    """Search vector database - backward compatibility function"""
    config = get_config().vector_db
    if model_name:
        config.model_name = model_name
    
    db = EnhancedVectorDatabase(config)
    return db.search_similar_chunks(query, database_path, top_k)