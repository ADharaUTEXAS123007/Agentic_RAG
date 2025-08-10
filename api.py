import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from main import AgenticRAG

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="A LangChain React agent with RAG capabilities for PDF documents and web search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG agent
agent = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class UploadResponse(BaseModel):
    message: str
    chunks_added: int

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG agent on startup"""
    global agent
    try:
        agent = AgenticRAG()
        print("RAG Agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RAG Agent: {e}")
        agent = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic RAG API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    return {"status": "healthy", "agent_ready": True}

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    if agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process the PDF
        result = agent.upload_pdf(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Extract number of chunks from result message
        chunks_added = 0
        if "Added" in result:
            try:
                chunks_added = int(result.split("Added")[1].split("chunks")[0].strip())
            except:
                pass
        
        return UploadResponse(
            message=result,
            chunks_added=chunks_added
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the RAG agent"""
    if agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    
    try:
        answer = agent.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/tools")
async def get_tools():
    """Get available tools"""
    if agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    
    tools_info = []
    for tool in agent.tools:
        tools_info.append({
            "name": tool.name,
            "description": tool.description
        })
    
    return {"tools": tools_info}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 