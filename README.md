# CorporateGPT Sparse-Reranker

Multi-Service API für RAG-Optimierung mit Sparse-Vektor-Generierung, PDF-Verarbeitung und Reranking.

## 🎯 Features

- **SPLADE v3 Sparse Vectors**: Hochwertige Sparse Embeddings für Hybrid Search
- **FlashRank Reranking**: Schnelles, ressourcenschonendes Reranking (~4-150MB Models)
- **PDF Processing**: URL-basierte Verarbeitung, Batch Processing, intelligentes Chunking
- **Production-Ready**: Docker-basiert, versioniert, reproduzierbar

---

## 🚀 Quick Start

### Voraussetzungen

- Docker & Docker Compose installiert
- Hugging Face API Token ([kostenlos erstellen](https://huggingface.co/settings/tokens))
- Mindestens 4 CPU Cores & 8 GB RAM

### Installation

```bash
git clone https://github.com/Sentrovo/CorporateGPT-Sparse-Reranker.git
cd CorporateGPT-Sparse-Reranker
nano .env
```

Im nano Editor folgendes einfügen:

```
HUGGINGFACE_API_KEY=hf_your_token_here
PORT=8000
MAX_WORKERS_PDF=10
MAX_WORKERS_SPARSE=2
FLASHRANK_MODEL=ms-marco-TinyBERT-L-2-v2
FLASHRANK_MAX_LENGTH=512
```


Service starten:

```bash
docker compose up -d
```

Health Check:

```bash
curl http://localhost:8000/health
```

**Fertig!** Service läuft auf `http://localhost:8000` 🎉

---

## 📡 API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Sparse Vector (Single)

```bash
curl -X POST http://localhost:8000/sparse \
  -H "Content-Type: application/json" \
  -d '{"text": "Was ist eine Vektordatenbank?"}'
```

### Sparse Vector (Batch)

```bash
curl -X POST http://localhost:8000/sparse/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "max_length": 512
  }'
```

### PDF Processing (URL)

```bash
curl -X POST http://localhost:8000/pdf/process-url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "file_name": "document.pdf",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

### PDF Batch Processing

```bash
curl -X POST http://localhost:8000/pdf/process-urls-batch \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      {"url": "https://example.com/doc1.pdf", "file_name": "doc1.pdf"},
      {"url": "https://example.com/doc2.pdf", "file_name": "doc2.pdf"}
    ]
  }'
```

### Reranking

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist eine Vektordatenbank?",
    "documents": [
      "{\"content\":\"Qdrant ist eine Vektordatenbank\"}",
      "{\"content\":\"PostgreSQL ist eine SQL Datenbank\"}"
    ],
    "top_n": 2
  }'
```

**API Dokumentation:** `http://localhost:8000/docs`

---

## 🛠️ Wartung

### Service stoppen

```bash
docker compose down
```

### Service neu bauen

```bash
docker compose up -d --build
```

### Logs anschauen

```bash
docker compose logs -f
```

### In Container einloggen

```bash
docker exec -it sparse-reranker bash
```

### Models Cache leeren

```bash
rm -rf models/
docker compose up -d --build
```

---

## 🔍 Troubleshooting

### CPU Error: `range of CPUs is from 0.01 to X.XX`

Passe `docker-compose.yml` an:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

```bash
docker compose up -d --build
```

### API Key Error

```bash
cat .env
```

Falls leer: `.env` neu erstellen mit deinem API Key.

### Service startet nicht

```bash
docker compose logs sparse-reranker
docker compose down
docker system prune -a
docker compose up -d --build
```

---

## 📚 Ressourcen

- **SPLADE v3**: https://huggingface.co/naver/splade-v3-lexical
- **FlashRank**: https://github.com/PrithivirajDamodaran/FlashRank
- **Qdrant Hybrid Search**: https://qdrant.tech/articles/sparse-vectors/

---

## 🤝 Support
- **Email**: info@sentrovo.de
