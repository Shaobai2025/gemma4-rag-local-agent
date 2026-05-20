# macOS 从零一键部署说明

本版默认加入 Ollama Embedding 模型拉取，默认模型为 embeddinggemma。

## 默认模型

```text
QA_MODEL=gemma4:e4b
NOTICE_MODEL=gemma4:e4b
PAPER_MODEL=gemma4:27b
EMBEDDING_MODEL=embeddinggemma
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

## 制作发布包

```bash
chmod +x install_mac_full.sh run_mac.sh doctor_mac.sh make_release_zip_mac_full.sh
bash make_release_zip_mac_full.sh
```

## 别人电脑部署

```bash
bash install_mac_full.sh
bash run_mac.sh
```

## 手动拉取模型

```bash
ollama pull gemma4:e4b
ollama pull gemma4:27b
ollama pull embeddinggemma
```

如果目标电脑不能拉取 gemma4 或 embeddinggemma，可部署时输入其他模型。
