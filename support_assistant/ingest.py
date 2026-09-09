"""Document Ingestion and Vector Storage Module.
Loads the 8 Zepto policy text documents, computes dense embeddings using local
sentence-transformers (all-MiniLM-L6-v2), and indexes them in a persistent ChromaDB
collection with cosine distance metric.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global lazy-loaded embedding model
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy load and return local SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_client(chroma_dir: Path = CHROMA_DIR) -> chromadb.ClientAPI:
    """Create and return persistent ChromaDB client."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_or_create_collection(client: chromadb.ClientAPI = None) -> chromadb.Collection:
    """Get or create the policy collection configured for cosine similarity."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Explicit cosine similarity
    )


def load_policy_documents(docs_dir: Path = DOCS_DIR) -> List[Tuple[str, str, str]]:
    """Read the 8 policy documents from docs directory.
    Returns: List of tuples (doc_id, title/topic, text_content).
    """
    doc_files = sorted(docs_dir.glob("doc_*.txt"))
    if len(doc_files) != 8:
        raise ValueError(f"Expected exactly 8 policy documents, found {len(doc_files)} in {docs_dir}")

    documents = []
    for f in doc_files:
        doc_id = f.stem  # e.g., 'doc_01'
        content = f.read_text(encoding="utf-8").strip()
        first_line = content.splitlines()[0] if content else "Zepto Policy"
        title = first_line.replace("Zepto ", "").replace(":", "").strip()
        documents.append((doc_id, title, content))

    return documents


def ingest_documents() -> int:
    """Ingest, embed, and store all policy documents into ChromaDB."""
    print("Starting document ingestion...")
    model = get_embedding_model()
    docs = load_policy_documents()

    client = get_chroma_client()
    # Delete existing collection if present to guarantee clean ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [d[0] for d in docs]
    metadatas = [{"title": d[1], "source_file": f"{d[0]}.txt"} for d in docs]
    texts = [d[2] for d in docs]

    print(f"Computing embeddings for {len(texts)} documents with {EMBEDDING_MODEL_NAME}...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    count = collection.count()
    print(f"Successfully ingested and indexed {count} policy documents in ChromaDB ('{COLLECTION_NAME}').")
    return count


if __name__ == "__main__":
    ingest_documents()
