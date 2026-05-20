import base64
import requests
from app.config import OLLAMA_BASE_URL, VISION_MODEL

def analyze_image_with_ollama(image_bytes: bytes, question: str) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "images": [image_b64],
                "content": question.strip() or "请详细分析这张图片的内容，并用中文回答。",
            }
        ],
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()
