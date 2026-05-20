from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from app.categories import load_categories

def infer_category(filename: str) -> str:
    name = filename.lower()
    keyword_map = {
        "discipline": ["纪律", "处分"],
        "insurance": ["医保", "报销"],
        "leave": ["请假", "归寝"],
        "scholarship": ["奖学金", "助学金", "评优"],
        "mental_health": ["心理"],
        "notice": ["通知"],
    }
    for key, words in keyword_map.items():
        if any(w in name for w in words):
            return key
    existing = {c["key"] for c in load_categories()}
    return "general" if "general" in existing else (next(iter(existing)) if existing else "general")

def load_single_file(path: str) -> List[Document]:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return PyPDFLoader(path).load()
    if lower.endswith(".docx"):
        return Docx2txtLoader(path).load()
    if lower.endswith(".md") or lower.endswith(".txt"):
        return TextLoader(path, encoding="utf-8").load()
    return []

def enrich_docs_metadata(docs: List[Document], filename: str, category: str) -> List[Document]:
    for idx, d in enumerate(docs):
        d.metadata["source_file"] = filename
        d.metadata["category"] = category
        d.metadata["doc_index"] = idx
    return docs
