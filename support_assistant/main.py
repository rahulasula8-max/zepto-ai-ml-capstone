"""FastAPI Application for Zepto Policy Support Assistant.
Exposes POST /ask endpoint powered by LangGraph RAG workflow with ChromaDB.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

try:
    from support_assistant.rag_graph import SupportResponse, ask_assistant
except ImportError:
    from rag_graph import SupportResponse, ask_assistant

app = FastAPI(
    title="Zepto Support Assistant API",
    description="Deterministic RAG Policy Assistant with ChromaDB & LangGraph",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str = Field(..., example="What is the delivery policy?")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=SupportResponse)
def ask_endpoint(request: AskRequest):
    """Process customer inquiry and return policy or general response."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        response = ask_assistant(query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
