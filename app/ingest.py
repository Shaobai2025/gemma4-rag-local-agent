import hashlib
import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import OLLAMA_BASE_URL, EMBED_MODEL, DOCS_DIR, CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from app.loaders import load_single_file, enrich_docs_metadata, infer_category
from app.store import load_registry, save_registry, read_sparse_rows, write_sparse_rows

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_vectorstore():
    return Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

def split_documents(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(raw_docs)

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_chunk_ids(filename: str, file_hash: str, docs: List[Document]) -> List[str]:
    ids = []
    for i, d in enumerate(docs):
        page_hash = hashlib.sha1(d.page_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
        ids.append(f"{filename}::{file_hash[:10]}::{i}::{page_hash}")
    return ids

def remove_file_indexes(filename: str) -> int:
    registry = load_registry()
    item = registry.get(filename)
    if not item:
        return 0
    chunk_ids = item.get("chunk_ids", [])
    if chunk_ids:
        vectorstore = get_vectorstore()
        vectorstore.delete(ids=chunk_ids)
    rows = read_sparse_rows()
    rows = [r for r in rows if r.get("metadata", {}).get("source_file") != filename]
    write_sparse_rows(rows)
    removed = len(chunk_ids)
    registry.pop(filename, None)
    save_registry(registry)
    return removed

def upsert_file(filepath: str, filename: str, category: str | None = None) -> Tuple[int, str, bool]:
    if category is None or not category.strip():
        category = infer_category(filename)
    current_hash = file_sha256(filepath)
    registry = load_registry()
    old = registry.get(filename)
    updated = old is not None
    if old and old.get("file_hash") == current_hash:
        return old.get("chunk_count", 0), category, updated
    if old:
        remove_file_indexes(filename)
    raw_docs = load_single_file(filepath)
    raw_docs = enrich_docs_metadata(raw_docs, filename, category)
    split_docs = split_documents(raw_docs)
    chunk_ids = build_chunk_ids(filename, current_hash, split_docs)
    for doc, cid in zip(split_docs, chunk_ids):
        doc.metadata["chunk_id"] = cid
        doc.metadata["file_hash"] = current_hash
    if split_docs:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(split_docs, ids=chunk_ids)
    rows = read_sparse_rows()
    rows.extend([{"page_content": d.page_content, "metadata": d.metadata} for d in split_docs])
    write_sparse_rows(rows)
    registry = load_registry()
    registry[filename] = {
        "filename": filename,
        "category": category,
        "file_hash": current_hash,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "size": os.path.getsize(filepath),
    }
    save_registry(registry)
    return len(chunk_ids), category, updated

def delete_file_and_indexes(filepath: str, filename: str) -> int:
    removed = remove_file_indexes(filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return removed

def bootstrap_existing_docs() -> Tuple[int, list[str]]:
    total = 0
    files = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(path):
            chunks, _, _ = upsert_file(path, fname)
            total += chunks
            files.append(fname)
    return total, files
