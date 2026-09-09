"""Automated Test Suite for Module 3 — Support Assistant.
Validates all graded criteria for document corpus, embeddings, ChromaDB, LangGraph
nodes and routing, Pydantic response models, and FastAPI POST /ask.
"""
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient

from ingest import DOCS_DIR, get_chroma_client, get_or_create_collection
from rag_graph import (
    POLICY_KEYWORDS,
    STRUCTURED_PROMPT_TEMPLATE,
    SupportResponse,
    ask_assistant,
    build_graph,
    classify_intent,
    direct_answer,
    retrieve_and_answer,
)
from main import app

client = TestClient(app)


def test_docs_corpus_exists():
    """Verify exactly 8 policy documents exist with expected non-empty content."""
    doc_files = sorted(DOCS_DIR.glob("doc_*.txt"))
    assert len(doc_files) == 8, f"Expected exactly 8 policy documents, found {len(doc_files)}"
    for f in doc_files:
        content = f.read_text(encoding="utf-8").strip()
        assert len(content) > 100, f"Document {f.name} has insufficient length ({len(content)} chars)"


def test_chroma_collection_populated():
    """Verify ChromaDB has zepto_policies collection containing all 8 documents."""
    c = get_chroma_client()
    col = get_or_create_collection(c)
    assert col.count() == 8, f"Expected 8 indexed documents in ChromaDB, found {col.count()}"


def test_structured_prompt_template():
    """Verify prompt template contains all 5 required elements, negative constraint, and few-shot."""
    p = STRUCTURED_PROMPT_TEMPLATE
    assert "Role:" in p, "Missing 'role' in prompt template"
    assert "Context:" in p, "Missing 'context' in prompt template"
    assert "Task:" in p, "Missing 'task' in prompt template"
    assert "Format:" in p, "Missing 'format' in prompt template"
    assert "Length:" in p, "Missing 'length' in prompt template"
    assert "Negative Constraint:" in p, "Missing negative constraint in prompt template"
    assert "Few-Shot Example:" in p, "Missing few-shot example in prompt template"


def test_classify_intent_keyword_heuristic():
    """Verify keyword heuristic routes policy terms to policy_question and others to general_question."""
    for kw in POLICY_KEYWORDS:
        res = classify_intent({"query": f"Tell me about {kw}"})
        assert res["intent"] == "policy_question", f"Keyword '{kw}' failed to classify as policy_question"

    res_gen = classify_intent({"query": "What is the capital of France?"})
    assert res_gen["intent"] == "general_question"


def test_retrieve_and_answer_mock_format():
    """Verify mock retrieval format 'Based on the retrieved context: {top_chunk_snippet}'."""
    state = {
        "query": "What is Zepto's delivery policy?",
        "intent": "policy_question",
        "retrieved_docs": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    }
    res = retrieve_and_answer(state)
    assert res["answer"].startswith("Based on the retrieved context:")
    assert len(res["sources"]) > 0
    assert "doc_01" in res["sources"], "Expected doc_01 (Delivery Policy) in sources"
    assert 0.0 <= res["confidence"] <= 1.0


def test_direct_answer_mock_format():
    """Verify deterministic canned general answer."""
    state = {
        "query": "Hello there",
        "intent": "general_question",
        "retrieved_docs": [],
        "sources": [],
        "answer": "",
        "confidence": 0.0
    }
    res = direct_answer(state)
    assert res["answer"] == "I can only answer questions about Zepto policies right now."
    assert res["sources"] == []
    assert res["confidence"] == 1.0


def test_fastapi_endpoints():
    """Verify FastAPI GET /health and POST /ask endpoints."""
    # Health check
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    # Policy Query
    r_pol = client.post("/ask", json={"query": "What is your return and refund policy?"})
    assert r_pol.status_code == 200
    data_pol = r_pol.json()
    assert "Based on the retrieved context:" in data_pol["answer"]
    assert "doc_02" in data_pol["sources"]
    assert 0.0 <= data_pol["confidence"] <= 1.0

    # General Query
    r_gen = client.post("/ask", json={"query": "How do I build a wooden chair?"})
    assert r_gen.status_code == 200
    data_gen = r_gen.json()
    assert data_gen["answer"] == "I can only answer questions about Zepto policies right now."
    assert data_gen["sources"] == []
    assert data_gen["confidence"] == 1.0

    print("\n--- RAW JSON API OUTPUTS (FOR README) ---")
    print("1. Policy / Retrieval Example (query: 'What is your return and refund policy?'):")
    print(json.dumps(data_pol, indent=2))
    print("\n2. General / Non-retrieval Example (query: 'How do I build a wooden chair?'):")
    print(json.dumps(data_gen, indent=2))


if __name__ == "__main__":
    print("Running test_docs_corpus_exists...")
    test_docs_corpus_exists()
    print("Running test_chroma_collection_populated...")
    test_chroma_collection_populated()
    print("Running test_structured_prompt_template...")
    test_structured_prompt_template()
    print("Running test_classify_intent_keyword_heuristic...")
    test_classify_intent_keyword_heuristic()
    print("Running test_retrieve_and_answer_mock_format...")
    test_retrieve_and_answer_mock_format()
    print("Running test_direct_answer_mock_format...")
    test_direct_answer_mock_format()
    print("Running test_fastapi_endpoints...")
    test_fastapi_endpoints()
    print("\nALL SUPPORT ASSISTANT TESTS PASSED!")
