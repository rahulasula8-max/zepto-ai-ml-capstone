# Module 3 — Support Assistant: Offline Policy RAG Engine

## Overview
The Zepto Support Assistant is an offline-first, deterministic, agentic retrieval-augmented generation (RAG) service designed to address customer inquiries regarding store operations, deliveries, returns, and subscription policies. The system is engineered around **LangGraph StateGraph**, local **Sentence-Transformers** dense embeddings, a persistent **ChromaDB** vector index, and **FastAPI**.

## RAG System Architecture

```text
                                  +-----------------------+
                                  |  User Query (POST /)  |
                                  +-----------+-----------+
                                              |
                                              v
                              +-------------------------------+
                              |    Node: classify_intent      |
                              |  (Keyword Heuristic / Router) |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     | Intent = "policy_question"                      | Intent = "general_question"
                     v                                                 v
   +------------------------------------+             +----------------------------------+
   |   Node: retrieve_and_answer        |             |       Node: direct_answer        |
   |                                    |             |                                  |
   | 1. Embed query locally             |             | Deterministic Canned Response:   |
   |    (all-MiniLM-L6-v2)              |             | "I can only answer questions     |
   | 2. Query ChromaDB (Cosine Sim)     |             |  about Zepto policies right now."|
   | 3. Retrieve TOP-3 Chunks           |             |                                  |
   | 4. Generation Branch:              |             | sources: []                      |
   |    - If MOCK_LLM != "0":           |             | confidence: 1.0                  |
   |      Template: "Based on..."       |             +-----------------+----------------+
   |    - If MOCK_LLM == "0":           |                               |
   |      Structured Prompt + Retry     |                               |
   +-----------------+------------------+                               |
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                               +-----------------------------+
                               |  Pydantic Validation Guard  |
                               |  - answer: str              |
                               |  - sources: list            |
                               |  - confidence: float (0..1) |
                               +--------------+--------------+
                                              |
                                              v
                               +-----------------------------+
                               |    HTTP 200 JSON Response   |
                               +-----------------------------+
```

### Architectural Stages
1. **Document Ingestion (`ingest.py`)**:
   - Manages exactly 8 canonical text files in `docs/`:
     - `doc_01.txt`: Delivery Policy
     - `doc_02.txt`: Returns & Refunds
     - `doc_03.txt`: Membership Tiers (Zepto Pass)
     - `doc_04.txt`: Order Tracking
     - `doc_05.txt`: Order Cancellation Policy
     - `doc_06.txt`: Damaged or Missing Items
     - `doc_07.txt`: Gift Cards
     - `doc_08.txt`: Customer Support Hours
   - Pre-chunks each document with preserved metadata (`title`, `source_file`).

2. **Embedding & Vector Indexing (`ingest.py` & `chroma_db/`)**:
   - Embeds text locally using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Stores embeddings into persistent ChromaDB collection **`zepto_policies`** with HNSW cosine distance (`space: cosine`).
   - Completely decoupled from external network access during runtime queries.

3. **Orchestration & Graph Routing (`rag_graph.py`)**:
   - Implemented via `langgraph.graph.StateGraph` utilizing an explicit `TypedDict` state:
     ```python
     class AgentState(TypedDict):
         query: str
         intent: str
         retrieved_docs: List[str]
         sources: List[str]
         answer: str
         confidence: float
     ```
   - Three named nodes:
     - `classify_intent`: Evaluates queries against policy keywords (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`).
     - `retrieve_and_answer`: Executes dense cosine retrieval of top 3 chunks and builds context-grounded response.
     - `direct_answer`: Handles non-policy general questions deterministically without calling retrieval.
   - Graph routing is conditional from `classify_intent` and does **not** branch on `MOCK_LLM`.

4. **Generation & MOCK_LLM Branching**:
   - **`MOCK_LLM=1` (or unset, Default Graded Baseline)**:
     - 100% offline and deterministic; requires zero API keys.
     - Policy answers follow the exact specification:
       `Based on the retrieved context: {top_chunk_snippet}`
       where `top_chunk_snippet` is the first ~200 characters of the top matching chunk.
     - Sources return retrieved document IDs (`['doc_01', ...]`).
     - Confidence is calculated from cosine similarity and normalized in `[0.0, 1.0]`.
     - General questions return: `"I can only answer questions about Zepto policies right now."`.
   - **`MOCK_LLM=0` (Optional Real LLM Mode)**:
     - Formulates structured prompt containing:
       - **Role**: Expert Zepto AI support assistant.
       - **Context**: Concatenated retrieved chunks.
       - **Task**: Accurate policy question answering.
       - **Format**: Valid JSON matching Pydantic response schema.
       - **Length**: Under 3 sentences.
       - **Negative Constraint**: *"Do not answer using information not present in the provided context."*
       - **Few-Shot Example**: Illustrative Q&A pair demonstration.
     - Incorporates automatic retry logic (up to 2 additional attempts) if output fails Pydantic schema validation.

5. **Pydantic Validation Guard**:
   - Enforces schema:
     ```python
     class SupportResponse(BaseModel):
         answer: str
         sources: List[str]
         confidence: float = Field(..., ge=0.0, le=1.0)
     ```

## Example API Calls (Raw JSON Output)

### 1. Retrieval / Policy Query
**Endpoint**: `POST /ask`  
**Request Payload**:
```json
{
  "query": "What is your return and refund policy?"
}
```
**Raw JSON Response**:
```json
{
  "answer": "Based on the retrieved context: Zepto Returns & Refunds Policy:\nCustomers may request returns or refunds for eligible items directly via the Zepto mobile app under the Order History section. For perishable items including fresh frui",
  "sources": [
    "doc_02",
    "doc_05",
    "doc_06"
  ],
  "confidence": 0.41
}
```

### 2. General / Non-Policy Query
**Endpoint**: `POST /ask`  
**Request Payload**:
```json
{
  "query": "How do I build a wooden chair?"
}
```
**Raw JSON Response**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Local & Docker Execution

### Local Python Execution
```bash
# 1. Populate/verify ChromaDB index
python support_assistant/ingest.py

# 2. Run test suite
python support_assistant/test_support_assistant.py

# 3. Launch FastAPI server via Uvicorn (port 7860)
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

### Docker Container Execution
```bash
# Build Docker image
docker build -t zepto-support-assistant -f support_assistant/Dockerfile support_assistant/

# Run container (port 7860 mapped)
docker run -d --name zepto-assistant -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant

# Verify container health
curl http://localhost:7860/health

# Send inquiry to container
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Zepto delivery policy?"}'
```
