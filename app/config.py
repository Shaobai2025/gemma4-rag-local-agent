import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = str(DATA_DIR / "docs")
CHROMA_DIR = str(DATA_DIR / "chroma_db")
STATE_DIR = DATA_DIR / "state"

FILE_REGISTRY_FILE = str(STATE_DIR / "file_registry.json")
SPARSE_INDEX_FILE = str(STATE_DIR / "sparse_rows.jsonl")
CATEGORIES_FILE = str(STATE_DIR / "categories.json")
HISTORY_FILE = str(STATE_DIR / "history.json")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

QA_MODEL = os.getenv("QA_MODEL", "gemma4:e4b")
NOTICE_MODEL = os.getenv("NOTICE_MODEL", "gemma4:e4b")
PAPER_MODEL = os.getenv("PAPER_MODEL", "gemma4:26b")
VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:e4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")

# 兼容旧代码
LLM_MODEL = QA_MODEL

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
HYBRID_DENSE_K = int(os.getenv("HYBRID_DENSE_K", "3"))
HYBRID_SPARSE_K = int(os.getenv("HYBRID_SPARSE_K", "3"))
HYBRID_FUSED_K = int(os.getenv("HYBRID_FUSED_K", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "gemma4_hybrid_rag")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
ALLOWED_ANALYSIS_EXTENSIONS = {".xlsx", ".xls", ".csv"}

DEFAULT_CATEGORIES = [
    {"key": "general", "label": "通用知识"},
    {"key": "leave", "label": "请假与归寝"},
    {"key": "insurance", "label": "医保与资助"},
    {"key": "discipline", "label": "纪律与处分"},
    {"key": "mental_health", "label": "心理健康"},
    {"key": "scholarship", "label": "奖助评优"},
    {"key": "notice", "label": "通知公告"},
]

for p in [DATA_DIR, Path(DOCS_DIR), Path(CHROMA_DIR), STATE_DIR]:
    p.mkdir(parents=True, exist_ok=True)