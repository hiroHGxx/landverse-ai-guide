"""
Enhanced Smart RAG with caching, better error handling, and performance optimizations
"""
import os
import hashlib
from functools import lru_cache
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import google.generativeai as genai

from .config import get_config
from .utils import get_logger, LoggerMixin, handle_exceptions
from .rag_core.enhanced_vector_db import EnhancedVectorDatabase


class TranslationCache:
    """Cache for translation results with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get(self, japanese_text: str) -> Optional[str]:
        """Get cached translation"""
        self._cleanup_expired()
        key = hashlib.md5(japanese_text.encode('utf-8')).hexdigest()
        
        if key in self.cache:
            translation, _ = self.cache[key]
            return translation
        return None
    
    def set(self, japanese_text: str, english_text: str):
        """Cache translation result"""
        self._cleanup_expired()
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = hashlib.md5(japanese_text.encode('utf-8')).hexdigest()
        self.cache[key] = (english_text, datetime.now())


class EnhancedSmartRAGSystem(LoggerMixin):
    """
    Enhanced Smart RAG system with caching, better error handling, and performance optimizations
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.vector_db = EnhancedVectorDatabase(self.config.vector_db)
        self.translation_cache = TranslationCache(
            max_size=self.config.llm.translation_cache_size
        )
        
        # Initialize Gemini API
        if self.config.llm.api_key:
            genai.configure(api_key=self.config.llm.api_key)
        else:
            self.logger.warning("Gemini API key not configured")
    
    @handle_exceptions()
    def translate_to_english(self, japanese_question: str) -> str:
        """Translate Japanese question to English with caching"""
        if not japanese_question.strip():
            return japanese_question
        
        # Check cache first
        cached_translation = self.translation_cache.get(japanese_question)
        if cached_translation:
            self.logger.debug(f"Using cached translation for: {japanese_question}")
            return cached_translation
        
        if not self.config.llm.api_key:
            self.logger.warning("No API key available for translation")
            return japanese_question
        
        try:
            model = genai.GenerativeModel(self.config.llm.model_name)
            
            prompt = f"""以下の日本語の質問を、自然で検索に適した英語に翻訳してください。
ゲーム関連の専門用語も考慮してください。翻訳結果のみを出力してください。

日本語の質問: {japanese_question}

英語の質問:"""
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.llm.max_tokens,
                    temperature=self.config.llm.temperature
                )
            )
            
            english_question = response.text.strip()
            
            # Cache the result
            self.translation_cache.set(japanese_question, english_question)
            
            self.logger.info(f"翻訳: 「{japanese_question}」 → 「{english_question}」")
            return english_question
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return japanese_question
    
    @handle_exceptions()
    def generate_japanese_response(self, context_chunks: list, original_japanese_question: str, 
                                 english_question: str) -> str:
        """Generate Japanese response using context chunks"""
        if not self.config.llm.api_key:
            return "❌ Gemini API key not configured."
        
        try:
            # Create context from chunks
            context = "\n\n".join([
                f"【参考情報 {i+1}】\n{chunk}" 
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Create prompt for Japanese response
            prompt = f"""以下の参考情報を元に、日本語の質問に日本語で回答してください。
参考情報は英語で書かれている場合がありますが、回答は必ず日本語で行ってください。
参考情報に含まれていない内容については推測で答えず、「参考情報には記載されていません」と回答してください。

参考情報：
{context}

元の質問（日本語）: {original_japanese_question}
検索に使用した質問（英語）: {english_question}

日本語での回答:"""
            
            model = genai.GenerativeModel(self.config.llm.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.llm.max_tokens,
                    temperature=self.config.llm.temperature
                )
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"❌ 回答生成エラー: {str(e)}"
    
    @handle_exceptions()
    def smart_query(self, japanese_question: str, top_k: int = 5) -> Dict:
        """Enhanced smart query with better error handling and logging"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Smart query started: {japanese_question}")
            
            # Step 1: Translate to English
            english_question = self.translate_to_english(japanese_question)
            
            # Step 2: Search vector database
            self.logger.debug(f"Searching with English query: {english_question}")
            search_results = self.vector_db.search_similar_chunks(
                english_question, 
                self.config.vector_db.database_path, 
                top_k
            )
            
            if not search_results:
                return {
                    "status": "error",
                    "message": "関連する情報が見つかりませんでした",
                    "original_question": japanese_question,
                    "translated_question": english_question,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Generate Japanese response
            context_chunks = [chunk for chunk, score in search_results]
            self.logger.info(f"Found {len(context_chunks)} relevant chunks")
            
            japanese_answer = self.generate_japanese_response(
                context_chunks, japanese_question, english_question
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "original_question": japanese_question,
                "translated_question": english_question,
                "answer": japanese_answer,
                "sources": [
                    {
                        "chunk_index": i,
                        "similarity_score": score,
                        "content_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    }
                    for i, (chunk, score) in enumerate(search_results)
                ],
                "context_chunks": context_chunks,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Smart query completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Smart query failed after {processing_time:.2f}s: {e}")
            
            return {
                "status": "error",
                "message": f"クエリ処理に失敗しました: {str(e)}",
                "original_question": japanese_question,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "translation_cache_size": len(self.translation_cache.cache),
            "translation_cache_max_size": self.translation_cache.max_size,
            "vector_db_stats": self.vector_db.get_stats()
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.translation_cache.cache.clear()
        self.vector_db._embedding_cache.clear()
        self.logger.info("All caches cleared")


# Convenience function for easy usage
@handle_exceptions()
def smart_ask(question: str, show_sources: bool = True, top_k: int = 5) -> str:
    """
    Convenience function for smart asking with enhanced features
    """
    logger = get_logger(__name__)
    
    try:
        smart_rag = EnhancedSmartRAGSystem()
        response = smart_rag.smart_query(question, top_k)
        
        if response["status"] == "error":
            logger.error(f"Smart ask failed: {response['message']}")
            return f"❌ エラー: {response['message']}"
        
        answer = response["answer"]
        
        if show_sources:
            print(f"\n📋 質問: {question}")
            print(f"🔄 翻訳: {response['translated_question']}")
            print(f"🤖 回答: {answer}")
            print(f"\n📚 ソース ({len(response['sources'])}個):")
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"  {i}. 類似度: {source['similarity_score']:.3f}")
            print(f"\n⏱️  処理時間: {response['processing_time']:.2f}秒")
            print()
        
        return answer
        
    except Exception as e:
        logger.error(f"Smart ask exception: {e}")
        return f"❌ システムエラー: {str(e)}"


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper