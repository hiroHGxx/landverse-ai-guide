import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .crawler.web_crawler import crawl_website
from .chunker.text_splitter import split_text_into_chunks
from .rag_core.vector_database import build_vector_database, search_vector_database


app = FastAPI(title="RAG AI Guide API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CRAWL_CONFIG = {
    "start_url": "https://guide.rolg.maxion.gg",
    "max_depth": 2,
    "domain": "guide.rolg.maxion.gg"
}
DATABASE_PATH = "data/vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY environment variable not set")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_chunks: List[str]


class BuildIndexResponse(BaseModel):
    message: str
    status: str


async def build_index_background():
    """Background task to build the vector database index."""
    try:
        print("Starting index building process...")
        
        # Step 1: Crawl website
        print(f"Crawling website: {CRAWL_CONFIG['start_url']}")
        crawled_data = crawl_website(
            start_url=CRAWL_CONFIG["start_url"],
            max_depth=CRAWL_CONFIG["max_depth"],
            domain=CRAWL_CONFIG["domain"]
        )
        
        if not crawled_data:
            print("No data was crawled from the website")
            return
        
        print(f"Crawled {len(crawled_data)} pages")
        
        # Step 2: Split text into chunks
        print("Splitting text into chunks...")
        all_chunks = []
        url_mapping = {}  # Map chunk index to source URL
        
        for url, text_content in crawled_data:
            if not text_content.strip():
                continue
                
            chunks = split_text_into_chunks(
                text_content,
                max_chunk_size=CHUNK_SIZE,
                overlap_size=CHUNK_OVERLAP
            )
            
            # Store URL mapping for each chunk
            for chunk in chunks:
                if chunk.strip():
                    url_mapping[len(all_chunks)] = url
                    all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} text chunks")
        
        if not all_chunks:
            print("No text chunks were created")
            return
        
        # Step 3: Build vector database
        print("Building vector database...")
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        
        build_vector_database(
            text_chunks=all_chunks,
            save_path=DATABASE_PATH
        )
        
        # Save URL mapping
        import pickle
        with open(f"{DATABASE_PATH}_url_mapping.pkl", 'wb') as f:
            pickle.dump(url_mapping, f)
        
        print("Index building completed successfully!")
        
    except Exception as e:
        print(f"Error during index building: {e}")
        raise


def generate_llm_response(context_chunks: List[str], question: str) -> str:
    """Generate response using Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Create context from chunks
        context = "\n\n".join([f"【参考情報 {i+1}】\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create prompt
        prompt = f"""あなたは親切で知識豊富な「Landverseのベテラン冒険者」です。以下の参考情報を元に、初心者の質問に対して、分かりやすく、丁寧な言葉で回答を生成してください。

重要なルール：
1. 参考情報に書かれていないことは、絶対に推測で答えないでください。その場合は「その情報は見つかりませんでした」と正直に回答してください。
2. 情報をただコピーするのではなく、あなた自身の言葉で要約し、必要であれば箇条書きなどを使って整理してください。
3. 回答は、まず結論から始め、その後に詳細を説明する構成にしてください。
4. マークダウン記号（**、*、#など）は一切使わず、プレーンテキストで回答してください。強調したい場合は「」（鍵括弧）を使用してください。

【参考情報】
{context}

【質問】
{question}

【回答】
"""
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG AI Guide API is running"}


@app.post("/api/build_index", response_model=BuildIndexResponse)
async def build_index(background_tasks: BackgroundTasks):
    """Build the vector database index from crawled website data."""
    try:
        # Add background task
        background_tasks.add_task(build_index_background)
        
        return BuildIndexResponse(
            message="Index building process started. This may take several minutes.",
            status="started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start index building: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system with a question."""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if database exists
        if not os.path.exists(f"{DATABASE_PATH}.faiss"):
            raise HTTPException(
                status_code=404, 
                detail="Vector database not found. Please build the index first using /api/build_index"
            )
        
        # Search for relevant chunks
        search_results = search_vector_database(
            query=question,
            database_path=DATABASE_PATH,
            top_k=5
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Extract chunks and their scores
        context_chunks = [chunk for chunk, score in search_results]
        
        # Load URL mapping
        url_mapping = {}
        try:
            import pickle
            with open(f"{DATABASE_PATH}_url_mapping.pkl", 'rb') as f:
                url_mapping = pickle.load(f)
        except FileNotFoundError:
            print("URL mapping file not found")
        
        # Generate LLM response
        answer = generate_llm_response(context_chunks, question)
        
        # Prepare sources information
        sources = []
        for i, (chunk, score) in enumerate(search_results):
            source_info = {
                "chunk_index": i,
                "similarity_score": score,
                "content_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "source_url": url_mapping.get(i, "Unknown")
            }
            sources.append(source_info)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            context_chunks=context_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)