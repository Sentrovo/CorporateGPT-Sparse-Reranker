# CorporateGPT Sparse-Reranker

Multi-Service API für RAG-Optimierung mit Sparse-Vektor-Generierung, PDF-Verarbeitung und Reranking.

## 🎯 Features

### 1. **SPLADE v3 Sparse Vectors**
- Generiert hochwertige Sparse Embeddings mit `naver/splade-v3-lexical`
- Single & Batch Processing
- GPU/CPU Support

### 2. **FlashRank Reranking**
- Schnelles, ressourcenschonendes Reranking (~4-150MB Models)
- Unterstützt mehrere Modelle (TinyBERT, MiniLM, MultiBERT)
- Optimiert für RAG-Pipelines

### 3. **PDF Processing**
- Upload-basierte Verarbeitung
- URL-basierter Download & Processing
- Batch Processing für mehrere PDFs parallel
- Intelligentes Chunking mit Page-Tracking

### 4. **Parallel Processing**
- ThreadPool-basierte Parallelverarbeitung
- Konfigurierbare Worker-Anzahl
- Optimiert für High-Throughput

---

## 🚀 Quick Start mit Docker

### Voraussetzungen

- Docker & Docker Compose installiert
- Hugging Face API Token ([kostenlos erstellen](https://huggingface.co/settings/tokens))
- Mindestens 8 GB RAM

### 1. Repository klonen
```bash
git clone https://github.com/Sentrovo/CorporateGPT-Sparse-Reranker.git
cd CorporateGPT-Sparse-Reranker
```

### 2. Umgebungsvariablen konfigurieren

Erstelle eine `.env` Datei:
```bash
# === REQUIRED ===
HUGGINGFACE_API_KEY=hf_your_token_here

# === OPTIONAL ===
PORT=8000
MAX_WORKERS_PDF=10
MAX_WORKERS_SPARSE=2
FLASHRANK_MODEL=ms-marco-TinyBERT-L-2-v2
FLASHRANK_MAX_LENGTH=512
```

### 3. Dockerfile erstellen

Erstelle ein `Dockerfile` im Root-Verzeichnis:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Application Code
COPY . .

# Model Cache Directory
RUN mkdir -p /models
ENV TRANSFORMERS_CACHE=/models

# Expose Port
EXPOSE 8000

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Service
CMD ["python", "main.py"]
```

### 4. Docker Compose Setup

Erstelle eine `docker-compose.yml`:
```yaml
version: '3.8'

services:
  sparse-reranker:
    build: .
    container_name: sparse-reranker
    ports:
      - "8000:8000"
    environment:
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
      - PORT=8000
      - MAX_WORKERS_PDF=10
      - MAX_WORKERS_SPARSE=2
      - FLASHRANK_MODEL=ms-marco-TinyBERT-L-2-v2
      - FLASHRANK_MAX_LENGTH=512
    volumes:
      - ./models:/models  # Model Cache persistent
      - ./logs:/app/logs  # Logs persistent
    restart: unless-stopped
    
    # Für GPU Support (NVIDIA):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
    
    # Resource Limits
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### 5. Service starten
```bash
# Build & Start
docker compose up -d

# Logs anzeigen
docker compose logs -f sparse-reranker

# Health Check
curl http://localhost:8000/health
```

---

## 📡 API Dokumentation

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "services": ["splade_sparse_vectors", "pdf_processing", "url_processing", "reranking"],
  "splade_model": {
    "status": "loaded",
    "model_id": "naver/splade-v3-lexical",
    "device": "cuda"
  },
  "flashrank_model": {
    "status": "loaded",
    "model_id": "ms-marco-TinyBERT-L-2-v2"
  },
  "version": "5.0.0-flashrank-integration"
}
```

---

### 1. Sparse Vectors

#### Single Text
```bash
POST /sparse
Content-Type: application/json

{
  "text": "Was ist eine Vektordatenbank?"
}
```

**Response:**
```json
{
  "indices": [123, 456, 789, 1234],
  "values": [0.85, 0.72, 0.68, 0.54],
  "model_used": "naver/splade-v3-lexical"
}
```

#### Batch Processing
```bash
POST /sparse/batch
Content-Type: application/json

{
  "texts": [
    "Erster Dokument-Text",
    "Zweiter Dokument-Text",
    "Dritter Dokument-Text"
  ],
  "max_length": 512
}
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [
    {"indices": [...], "values": [...]},
    {"indices": [...], "values": [...]},
    {"indices": [...], "values": [...]}
  ],
  "model_used": "naver/splade-v3-lexical"
}
```

---

### 2. PDF Processing

#### Upload PDF (Legacy)
```bash
POST /pdf/extract-and-chunk
Content-Type: multipart/form-data

file: document.pdf
chunk_size: 1000
chunk_overlap: 200
```

#### Process PDF from URL (Empfohlen)
```bash
POST /pdf/process-url
Content-Type: application/json

{
  "url": "https://example.com/document.pdf",
  "file_name": "document.pdf",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Response:**
```json
{
  "success": true,
  "file_name": "document.pdf",
  "source_url": "https://example.com/document.pdf",
  "download_size_mb": 2.5,
  "extraction": {
    "total_pages": 15,
    "total_chars": 45000,
    "total_words": 7500
  },
  "chunking": {
    "total_chunks": 35,
    "chunks": [
      {
        "chunk_index": 0,
        "text": "Chunk text...",
        "char_start": 0,
        "char_end": 1000,
        "primary_page": 1,
        "page_numbers": [1],
        "page_coverage": {"1": 1.0}
      }
    ]
  },
  "summary": {
    "total_pages": 15,
    "total_chunks": 35,
    "avg_chunks_per_page": 2.33
  }
}
```

#### Batch PDF Processing from URLs
```bash
POST /pdf/process-urls-batch
Content-Type: application/json

{
  "files": [
    {
      "url": "https://example.com/doc1.pdf",
      "file_name": "doc1.pdf",
      "chunk_size": 1000,
      "chunk_overlap": 200
    },
    {
      "url": "https://example.com/doc2.pdf",
      "file_name": "doc2.pdf",
      "chunk_size": 1000,
      "chunk_overlap": 200
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "success": true,
      "file_name": "doc1.pdf",
      "batch_index": 0,
      ...
    },
    {
      "success": true,
      "file_name": "doc2.pdf",
      "batch_index": 1,
      ...
    }
  ]
}
```

---

### 3. Reranking
```bash
POST /rerank
Content-Type: application/json

{
  "query": "Was ist eine Vektordatenbank?",
  "documents": [
    "{\"title\":\"Qdrant\",\"content\":\"Qdrant ist eine Vektordatenbank...\",\"page\":\"1\",\"file\":\"doc.pdf\"}",
    "{\"title\":\"PostgreSQL\",\"content\":\"PostgreSQL ist eine relationale DB...\",\"page\":\"2\",\"file\":\"doc.pdf\"}",
    "{\"title\":\"Embeddings\",\"content\":\"Vektordatenbanken speichern Embeddings...\",\"page\":\"3\",\"file\":\"doc.pdf\"}"
  ],
  "top_n": 2,
  "max_tokens_per_doc": 4096
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95
    },
    {
      "index": 2,
      "relevance_score": 0.87
    }
  ]
}
```

---


## 📚 Weitere Ressourcen

- **SPLADE v3**: https://huggingface.co/naver/splade-v3-lexical
- **FlashRank**: https://github.com/PrithivirajDamodaran/FlashRank
- **Qdrant Hybrid Search**: https://qdrant.tech/articles/sparse-vectors/
- **PyMuPDF**: https://pymupdf.readthedocs.io/

---

## 🤝 Support

Bei Fragen oder Problemen:
- **GitHub Issues**: https://github.com/Sentrovo/CorporateGPT-Sparse-Reranker/issues
- **Email**: info@sentrovo.de
