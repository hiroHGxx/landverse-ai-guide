import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional


class VectorDatabase:
    """
    Vector database for storing and searching text chunks using FAISS and sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector database with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []
        self.dimension = None
    
    def build_database(self, text_chunks: List[str], save_path: str) -> None:
        """
        Build a vector database from text chunks and save to file.
        
        Args:
            text_chunks: List of text chunks to index
            save_path: Path to save the database files (without extension)
        """
        if not text_chunks:
            raise ValueError("Text chunks list cannot be empty")
        
        print(f"Encoding {len(text_chunks)} text chunks...")
        
        # Convert text chunks to vectors
        embeddings = self.model.encode(text_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Get embedding dimension
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store text chunks
        self.text_chunks = text_chunks
        
        # Save to files
        self._save_database(save_path)
        
        print(f"Vector database built and saved to {save_path}")
    
    def search_similar_chunks(self, query: str, database_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar text chunks given a query.
        
        Args:
            query: The search query
            database_path: Path to the saved database files (without extension)
            top_k: Number of top similar chunks to return
            
        Returns:
            List of tuples containing (text_chunk, similarity_score)
        """
        # Load database if not already loaded
        if self.index is None or not self.text_chunks:
            self._load_database(database_path)
        
        if self.index is None:
            raise ValueError(f"Could not load database from {database_path}")
        
        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.text_chunks)))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((self.text_chunks[idx], float(score)))
        
        return results
    
    def _save_database(self, save_path: str) -> None:
        """
        Save the FAISS index and metadata to files.
        
        Args:
            save_path: Base path for saving files (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.faiss")
        
        # Save metadata (text chunks and model info)
        metadata = {
            'text_chunks': self.text_chunks,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    def _load_database(self, database_path: str) -> None:
        """
        Load the FAISS index and metadata from files.
        
        Args:
            database_path: Base path for loading files (without extension)
        """
        faiss_path = f"{database_path}.faiss"
        metadata_path = f"{database_path}.pkl"
        
        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Database files not found at {database_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.text_chunks = metadata['text_chunks']
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        
        # Ensure we have the correct model loaded
        if not hasattr(self, 'model') or self.model is None:
            self.model = SentenceTransformer(self.model_name)


def build_vector_database(text_chunks: List[str], save_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Convenience function to build a vector database from text chunks.
    
    Args:
        text_chunks: List of text chunks to index
        save_path: Path to save the database files (without extension)
        model_name: Name of the sentence-transformers model to use
    """
    db = VectorDatabase(model_name)
    db.build_database(text_chunks, save_path)


def search_vector_database(query: str, database_path: str, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[str, float]]:
    """
    Convenience function to search a vector database.
    
    Args:
        query: The search query
        database_path: Path to the saved database files (without extension)
        top_k: Number of top similar chunks to return
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        List of tuples containing (text_chunk, similarity_score)
    """
    db = VectorDatabase(model_name)
    return db.search_similar_chunks(query, database_path, top_k)