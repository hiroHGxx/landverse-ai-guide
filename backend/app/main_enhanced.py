"""
Enhanced main application with configuration management and improved architecture
"""
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .config import get_config, reload_config
from .utils import get_logger, RAGLogger
from .crawler.enhanced_crawler import EnhancedWebCrawler
from .chunker.text_splitter import split_text_into_chunks
from .rag_core.enhanced_vector_db import EnhancedVectorDatabase
from .smart_rag_enhanced import EnhancedSmartRAGSystem


# Initialize logging first
RAGLogger.setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting RAG AI Guide API")
    
    # Load configuration
    config = get_config()
    logger.info(f"Configuration loaded from: {config}")
    
    # Any startup tasks can go here
    yield
    
    # Cleanup tasks
    logger.info("Shutting down RAG AI Guide API")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Enhanced RAG AI Guide API",
    version="2.0.0",
    description="Production-ready RAG system with advanced features",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
smart_rag_system: EnhancedSmartRAGSystem = None


# Pydantic models
class BuildIndexRequest(BaseModel):
    """Request model for building index"""
    start_url: str = Field(default=None, description="Starting URL for crawling")
    max_depth: int = Field(default=None, ge=0, le=5, description="Maximum crawl depth")
    domain: str = Field(default=None, description="Domain to restrict crawling to")


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str = Field(..., min_length=1, description="Question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to return")
    show_translation: bool = Field(default=True, description="Show translation process")


class QueryResponse(BaseModel):
    """Response model for queries"""
    status: str
    answer: str
    original_question: str
    translated_question: str = None
    sources: List[Dict[str, Any]]
    processing_time: float
    timestamp: str


class BuildIndexResponse(BaseModel):
    """Response model for index building"""
    message: str
    status: str
    task_id: str = None


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    vector_db_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    config_summary: Dict[str, Any]


def get_smart_rag_system() -> EnhancedSmartRAGSystem:
    """Get or create smart RAG system instance"""
    global smart_rag_system
    if smart_rag_system is None:
        smart_rag_system = EnhancedSmartRAGSystem()
        logger.info("Smart RAG system initialized")
    return smart_rag_system


async def build_index_background(build_request: BuildIndexRequest, task_id: str):
    """Background task to build the vector database index"""
    try:
        logger.info(f"Starting index building task {task_id}")
        config = get_config()
        
        # Use request parameters or fall back to config
        start_url = build_request.start_url or config.crawler.start_url
        max_depth = build_request.max_depth if build_request.max_depth is not None else config.crawler.max_depth
        domain = build_request.domain or config.crawler.domain
        
        logger.info(f"Crawling {start_url} with depth {max_depth} for domain {domain}")
        
        # Step 1: Crawl website using enhanced crawler
        async with EnhancedWebCrawler(config.crawler) as crawler:
            crawled_data = await crawler.crawl_website(start_url, max_depth, domain)
        
        if not crawled_data:
            logger.error("No data was crawled from the website")
            return {"status": "error", "message": "No data crawled"}
        
        logger.info(f"Crawled {len(crawled_data)} pages")
        
        # Step 2: Split text into chunks
        logger.info("Processing text chunks")
        all_chunks = []
        url_mapping = {}
        
        for url, text_content in crawled_data:
            if not text_content.strip():
                continue
                
            chunks = split_text_into_chunks(
                text_content,
                max_chunk_size=config.chunker.max_chunk_size,
                overlap_size=config.chunker.overlap_size
            )
            
            for chunk in chunks:
                if chunk.strip():
                    url_mapping[len(all_chunks)] = url
                    all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} text chunks")
        
        if not all_chunks:
            logger.error("No text chunks were created")
            return {"status": "error", "message": "No chunks created"}
        
        # Step 3: Build enhanced vector database
        logger.info("Building enhanced vector database")
        vector_db = EnhancedVectorDatabase(config.vector_db)
        vector_db.build_database(all_chunks, config.vector_db.database_path)
        
        # Save URL mapping
        import pickle
        import os
        os.makedirs(os.path.dirname(config.vector_db.database_path), exist_ok=True)
        with open(f"{config.vector_db.database_path}_url_mapping.pkl", 'wb') as f:
            pickle.dump(url_mapping, f)
        
        logger.info(f"Index building task {task_id} completed successfully")
        return {
            "status": "success",
            "message": f"Built index with {len(all_chunks)} chunks from {len(crawled_data)} pages",
            "chunks_count": len(all_chunks),
            "pages_count": len(crawled_data)
        }
        
    except Exception as e:
        logger.error(f"Index building task {task_id} failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Enhanced RAG AI Guide API is running",
        "version": "2.0.0",
        "status": "healthy"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    try:
        config = get_config()
        return {
            "status": "healthy",
            "api_configured": bool(config.llm.api_key),
            "database_configured": bool(config.vector_db.database_path),
            "timestamp": logger._initialized  # Using logger as a proxy for startup time
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/api/build_index", response_model=BuildIndexResponse, tags=["Index Management"])
async def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """Build the vector database index from crawled website data"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # Add background task
        background_tasks.add_task(build_index_background, request, task_id)
        
        logger.info(f"Index building task {task_id} started")
        
        return BuildIndexResponse(
            message="Index building process started. This may take several minutes.",
            status="started",
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Failed to start index building: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start index building: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Query the enhanced RAG system with a question"""
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Get smart RAG system
        smart_rag = get_smart_rag_system()
        
        # Process query
        response = smart_rag.smart_query(request.question, request.top_k)
        
        if response["status"] == "error":
            logger.warning(f"Query failed: {response['message']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=response["message"]
            )
        
        logger.info(f"Query processed successfully in {response['processing_time']:.2f}s")
        
        return QueryResponse(
            status=response["status"],
            answer=response["answer"],
            original_question=response["original_question"],
            translated_question=response.get("translated_question") if request.show_translation else None,
            sources=response["sources"],
            processing_time=response["processing_time"],
            timestamp=response["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/api/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats():
    """Get system statistics and cache information"""
    try:
        smart_rag = get_smart_rag_system()
        config = get_config()
        
        stats = SystemStatsResponse(
            vector_db_stats=smart_rag.vector_db.get_stats(),
            cache_stats=smart_rag.get_cache_stats(),
            config_summary={
                "crawler_domain": config.crawler.domain,
                "chunk_size": config.chunker.max_chunk_size,
                "vector_model": config.vector_db.model_name,
                "llm_model": config.llm.model_name,
                "database_path": config.vector_db.database_path
            }
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system statistics"
        )


@app.post("/api/clear_cache", tags=["System"])
async def clear_cache():
    """Clear all system caches"""
    try:
        smart_rag = get_smart_rag_system()
        smart_rag.clear_caches()
        
        logger.info("System caches cleared")
        return {"message": "All caches cleared successfully", "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear caches"
        )


@app.post("/api/reload_config", tags=["System"])
async def reload_system_config():
    """Reload system configuration"""
    try:
        config = reload_config()
        
        # Reinitialize global systems with new config
        global smart_rag_system
        smart_rag_system = None  # Will be recreated with new config
        
        logger.info("Configuration reloaded successfully")
        return {
            "message": "Configuration reloaded successfully",
            "status": "success",
            "config_summary": {
                "crawler_domain": config.crawler.domain,
                "vector_model": config.vector_db.model_name,
                "llm_model": config.llm.model_name
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload configuration"
        )


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    logger.info("Starting Enhanced RAG AI Guide API server")
    
    uvicorn.run(
        "app.main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )