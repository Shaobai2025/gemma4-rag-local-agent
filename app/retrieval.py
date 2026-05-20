import json
import os
from typing import Dict, List, Optional, Tuple
import jieba
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.config import (
    OLLAMA_BASE_URL, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME,
    SPARSE_INDEX_FILE, HYBRID_DENSE_K, HYBRID_SPARSE_K, HYBRID_FUSED_K, RRF_K
)

def jieba_preprocess(text: str) -> List[str]:
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_vectorstore() -> Chroma:
    return Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

def load_sparse_docs() -> List[Document]:
    docs = []
    if not os.path.exists(SPARSE_INDEX_FILE):
        return docs
    with open(SPARSE_INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(Document(page_content=row["page_content"], metadata=row.get("metadata", {})))
    return docs

def dense_retrieve(question: str, category: Optional[str] = None) -> List[Document]:
    try:
        if not os.path.exists(CHROMA_DIR):
            return []
        vectorstore = get_vectorstore()
        if category:
            return vectorstore.similarity_search(question, k=HYBRID_DENSE_K, filter={"category": category})
        return vectorstore.similarity_search(question, k=HYBRID_DENSE_K)
    except Exception as e:
        print(f"[WARN] dense retrieve failed: {e}")
        return []

def sparse_retrieve(question: str, category: Optional[str] = None) -> List[Document]:
    try:
        docs = load_sparse_docs()
        if category:
            docs = [d for d in docs if d.metadata.get("category") == category]
        if not docs:
            return []
        retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocess)
        retriever.k = HYBRID_SPARSE_K
        return retriever.invoke(question)
    except Exception as e:
        print(f"[WARN] sparse retrieve failed: {e}")
        return []

def doc_key(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunk_id", ""))

def reciprocal_rank_fusion(
    dense_docs: List[Document], sparse_docs: List[Document], weights: Tuple[float, float] = (0.6, 0.4)
) -> List[Document]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs, start=1):
        key = doc_key(doc)
        if not key:
            continue
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + weights[0] * (1.0 / (RRF_K + rank))
    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc_key(doc)
        if not key:
            continue
        doc_map[key] = doc
        scores[key] = scores.get(key, 0.0) + weights[1] * (1.0 / (RRF_K + rank))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:HYBRID_FUSED_K]]

def hybrid_retrieve(question: str, category: Optional[str] = None) -> List[Document]:
    dense_docs = dense_retrieve(question, category)
    sparse_docs = sparse_retrieve(question, category)
    if not dense_docs and not sparse_docs:
        return []
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
    if not fused:
        fallback = []
        seen = set()
        for doc in dense_docs + sparse_docs:
            key = id(doc)
            if key in seen:
                continue
            seen.add(key)
            fallback.append(doc)
        return fallback[:HYBRID_FUSED_K]
    return fused
