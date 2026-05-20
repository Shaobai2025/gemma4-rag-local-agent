#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PROJECT_NAME="$(basename "$PROJECT_DIR")"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_DIR/release"
OUT_ZIP="$OUT_DIR/${PROJECT_NAME}_mac_full_deploy_gemma4_embeddinggemma_${DATE_TAG}.zip"

mkdir -p "$OUT_DIR"

echo "===================================================="
echo "  制作给别人电脑从零部署的发布包"
echo "===================================================="
echo "默认模型：gemma4:e4b / gemma4:27b"
echo "默认 Embedding：embeddinggemma"
echo "输出文件：$OUT_ZIP"
echo ""

zip -r "$OUT_ZIP" .   -x ".venv/*"   -x "__pycache__/*"   -x "*/__pycache__/*"   -x ".git/*"   -x ".DS_Store"   -x ".env"   -x "data/excel_sessions/*"   -x "data/uploads/*"   -x "data/chroma/*"   -x "data/vectorstore/*"   -x "docs/*"   -x "uploads/*"   -x "logs/*"   -x "release/*"   -x "*.log"

echo ""
echo "✅ 发布包已生成："
echo "$OUT_ZIP"
echo ""
echo "别人解压后执行："
echo "  bash install_mac_full.sh"
echo "  bash run_mac.sh"
echo "===================================================="
