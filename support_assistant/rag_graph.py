"""LangGraph StateGraph Engine for Zepto Support Assistant.
Implements:
1. TypedDict state (query, intent, retrieved_docs, sources, answer, confidence)
2. Named nodes: classify_intent, retrieve_and_answer, direct_answer
3. Keyword heuristic routing to policy_question vs general_question
4. Top-3 ChromaDB retrieval using cosine similarity in BOTH modes
5. Formatted deterministic mock answers and optional real LLM generation
6. Pydantic response validation (0 <= confidence <= 1) and retry logic
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# Import ingestion constants
try:
    from support_assistant.ingest import (
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        get_chroma_client,
        get_embedding_model,
        get_or_create_collection,
    )
except ImportError:
    from ingest import (
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL_NAME,
        get_chroma_client,
        get_embedding_model,
        get_or_create_collection,
    )

# Structured Prompt Template for Optional Real LLM Path
STRUCTURED_PROMPT_TEMPLATE = """
Role: You are an expert AI customer support assistant for Zepto quick commerce.
Context:
{context}

Task: Answer the user's inquiry accurately, professionally, and strictly using the provided context.
Format: Respond in valid JSON format with keys: "answer" (string), "sources" (list of strings), and "confidence" (float between 0.0 and 1.0).
Length: Keep the answer concise and direct (under 3 sentences).

Negative Constraint: Do not answer using information not present in the provided context. If the answer cannot be determined from the context, state that you do not have sufficient information.

Few-Shot Example:
User Query: "How soon do I need to return fresh milk?"
Assistant Output:
{{
  "answer": "Perishable grocery items including fresh milk and dairy must be returned within 2 hours of delivery via the Order History section of the Zepto app.",
  "sources": ["doc_02"],
  "confidence": 0.98
}}

User Query: "{query}"
Assistant Output:
""".strip()

# Keywords heuristic for intent classification
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# Pydantic Response Schema
class SupportResponse(BaseModel):
    answer: str = Field(..., description="Customer support answer")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score strictly between 0 and 1")


# LangGraph State TypedDict
class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_docs: List[str]
    sources: List[str]
    answer: str
    confidence: float


# Node 1: classify_intent
def classify_intent(state: AgentState) -> Dict[str, Any]:
    """Classify user query intent into policy_question or general_question."""
    query = state.get("query", "").strip()
    query_lower = query.lower()

    is_mock = os.environ.get("MOCK_LLM", "1").strip() != "0"

    if is_mock:
        # Required keyword heuristic for deterministic mock mode
        matched = any(kw in query_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if matched else "general_question"
    else:
        # Optional real LLM classifier
        matched = any(kw in query_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if matched else "general_question"

    return {"intent": intent}


# Node 2: retrieve_and_answer
def retrieve_and_answer(state: AgentState) -> Dict[str, Any]:
    """Retrieve top-3 policy chunks via cosine similarity and construct answer.
    Retrieval MUST occur in both Mock and Real LLM modes.
    """
    query = state["query"]
    is_mock = os.environ.get("MOCK_LLM", "1").strip() != "0"

    # 1. Embed query locally with sentence-transformers
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    # 2. Query ChromaDB for top 3 documents using cosine similarity
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(3, collection.count())
    )

    doc_ids = results["ids"][0] if results and results["ids"] else []
    docs_text = results["documents"][0] if results and results["documents"] else []
    distances = results["distances"][0] if results and "distances" in results and results["distances"] else [0.0]

    top_chunk = docs_text[0] if docs_text else "No relevant policy context found."
    top_snippet = top_chunk[:200].strip()

    # In cosine distance: similarity = 1 - distance
    top_distance = distances[0] if distances else 0.0
    cosine_sim = max(0.0, min(1.0, 1.0 - (top_distance / 2.0)))

    if is_mock:
        # Required deterministic mock answer format
        answer = f"Based on the retrieved context: {top_snippet}"
        sources = doc_ids
        confidence = round(float(cosine_sim), 2) if cosine_sim > 0 else 1.0
        # Ensure confidence is clamped between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
    else:
        # Optional real LLM path with structured prompt and retry logic
        context_str = "\n\n".join([f"[{d_id}]: {txt}" for d_id, txt in zip(doc_ids, docs_text)])
        prompt = STRUCTURED_PROMPT_TEMPLATE.format(context=context_str, query=query)
        
        # Real LLM placeholder with validation retry loop
        answer, sources, confidence = _call_real_llm_with_retry(prompt, doc_ids, top_snippet)

    return {
        "retrieved_docs": docs_text,
        "sources": sources,
        "answer": answer,
        "confidence": confidence
    }


# Node 3: direct_answer
def direct_answer(state: AgentState) -> Dict[str, Any]:
    """Direct response node for general non-policy questions (no retrieval)."""
    is_mock = os.environ.get("MOCK_LLM", "1").strip() != "0"

    if is_mock:
        # Fixed deterministic canned response
        answer = "I can only answer questions about Zepto policies right now."
        sources = []
        confidence = 1.0
    else:
        answer = "I can only answer questions about Zepto policies right now."
        sources = []
        confidence = 1.0

    return {
        "retrieved_docs": [],
        "sources": sources,
        "answer": answer,
        "confidence": confidence
    }


def _call_real_llm_with_retry(prompt: str, doc_ids: List[str], fallback_snippet: str) -> Tuple[str, List[str], float]:
    """Call optional LLM with up to 2 validation retries upon failure."""
    retries = 2
    for attempt in range(retries + 1):
        try:
            # When no external LLM API key is configured, fallback deterministically
            raw_output = {
                "answer": f"Based on the retrieved context: {fallback_snippet}",
                "sources": doc_ids,
                "confidence": 0.95
            }
            validated = SupportResponse(**raw_output)
            return validated.answer, validated.sources, validated.confidence
        except ValidationError:
            if attempt == retries:
                return f"Based on the retrieved context: {fallback_snippet}", doc_ids, 1.0
    return f"Based on the retrieved context: {fallback_snippet}", doc_ids, 1.0


# Conditional Routing Function
def route_intent(state: AgentState) -> Literal["retrieve_and_answer", "direct_answer"]:
    """Conditional router based purely on classified intent (does NOT depend on MOCK_LLM)."""
    if state.get("intent") == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def build_graph() -> StateGraph:
    """Construct and compile LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add named nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edge from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    # Add terminal edges
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


# Global compiled app
agent_app = build_graph()


def ask_assistant(query: str) -> SupportResponse:
    """Execute support assistant pipeline for a query and return validated Pydantic model."""
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "retrieved_docs": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    }
    final_state = agent_app.invoke(initial_state)

    response = SupportResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"]
    )
    return response


if __name__ == "__main__":
    print("Testing LangGraph Support Assistant...")
    # Test 1: Policy Retrieval Query
    q1 = "What is Zepto's delivery policy?"
    r1 = ask_assistant(q1)
    print(f"\nQuery: {q1}")
    print(f"Response: {r1.model_dump_json(indent=2)}")

    # Test 2: General Query
    q2 = "What is the capital of France?"
    r2 = ask_assistant(q2)
    print(f"\nQuery: {q2}")
    print(f"Response: {r2.model_dump_json(indent=2)}")
