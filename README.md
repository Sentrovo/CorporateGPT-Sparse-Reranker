# CorporateGPT Sparse-Reranker

Multi-Service API für RAG-Optimierung mit Sparse-Vektor-Generierung, PDF-Verarbeitung und Reranking.

## 🎯 Features

- **SPLADE v3 Sparse Vectors**: Hochwertige Sparse Embeddings für Hybrid Search (`naver/splade-v3-lexical`)
- **FlashRank Reranking**: Schnelles, ressourcenschonendes Reranking (~4-150MB Models)
- **PDF Processing**: URL-basierte Verarbeitung, Batch Processing, intelligentes Chunking mit Page-Tracking
- **Parallel Processing**: ThreadPool-basiert, konfigurierbare Worker-Anzahl, optimiert für High-Throughput
- **Production-Ready**: Docker-basiert, versioniert, reproduzierbar

---

## 🚀 Quick Start (3 Schritte)

### Voraussetzungen

- Docker & Docker Compose installiert ([Installation Guide](https://docs.docker.com/get-docker/))
- Hugging Face API Token ([kostenlos erstellen](https://huggingface.co/settings/tokens))
- **Mindestens 4 CPU Cores**
- **Mindestens 8 GB RAM**

### Installation

```bash
# 1. Repository klonen
git clone https://github.com/Sentrovo/CorporateGPT-Sparse-Reranker.git
cd CorporateGPT-Sparse-Reranker

# 2. Environment konfigurieren
cp .env.example .env
nano .env  # Trage deinen HUGGINGFACE_API_KEY ein, dann Ctrl+O, Enter, Ctrl+X

# 3. Service starten
docker compose up -d
```

### Verifizieren

```bash
# Logs anschauen
docker compose logs -f

# Health Check (in neuem Terminal)
curl http://localhost:8000/health
```

**Fertig!** Der Service läuft auf `http://localhost:8000` 🎉

**Erwartete Response:**
```json
{
  "status": "healthy",
  "splade_model": {"status": "loaded", "model_id": "naver/splade-v3-lexical"},
  "flashrank_model": {"status": "loaded", "model_id": "ms-marco-TinyBERT-L-2-v2"}
}
```

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Sparse Vectors (Single)
```bash
curl -X POST http://localhost:8000/sparse \
  -H "Content-Type: application/json" \
  -d '{"text": "Was ist eine Vektordatenbank?"}'
```

**Response:**
```json
{
  "indices": [123, 456, 789, 1234],
  "values": [0.85, 0.72, 0.68, 0.54],
  "model_used": "naver/splade-v3-lexical"
}
```

### Sparse Vectors (Batch)
```bash
curl -X POST http://localhost:8000/sparse/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Erster Dokument-Text",
      "Zweiter Dokument-Text",
      "Dritter Dokument-Text"
    ],
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

**Response:**
```json
{
  "success": true,
  "file_name": "document.pdf",
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
  }
}
```

### PDF Processing (Batch)
```bash
curl -X POST http://localhost:8000/pdf/process-urls-batch \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Reranking
```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist eine Vektordatenbank?",
    "documents": [
      "{\"title\":\"Qdrant\",\"content\":\"Qdrant ist eine Vektordatenbank...\"}",
      "{\"title\":\"PostgreSQL\",\"content\":\"PostgreSQL ist eine relationale DB...\"}",
      "{\"title\":\"Embeddings\",\"content\":\"Vektordatenbanken speichern Embeddings...\"}"
    ],
    "top_n": 2,
    "max_tokens_per_doc": 4096
  }'
```

**Response:**
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.87}
  ]
}
```

**Vollständige API-Dokumentation:** `http://localhost:8000/docs` (automatisch generiert)

---

## 🔧 Konfiguration

Alle Einstellungen über `.env` File:

```bash
# === REQUIRED ===
HUGGINGFACE_API_KEY=your_token_here

# === OPTIONAL (Defaults) ===
PORT=8000
MAX_WORKERS_PDF=10
MAX_WORKERS_SPARSE=2
FLASHRANK_MODEL=ms-marco-TinyBERT-L-2-v2
FLASHRANK_MAX_LENGTH=512
```

---

## 🛠️ Entwicklung & Wartung

### Service stoppen
```bash
docker compose down
```

### Service neu bauen (nach Code-Änderungen)
```bash
docker compose up -d --build
```

### Logs live anschauen
```bash
docker compose logs -f sparse-reranker
```

### In Container einloggen (Debugging)
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

### Error: `range of CPUs is from 0.01 to X.XX`

**Problem:** Dein Server hat weniger als 4 CPUs.

**Lösung:** Passe `docker-compose.yml` an:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'      # ← Ändere auf deine CPU-Anzahl
      memory: 4G     # ← Ggf. auch RAM reduzieren
    reservations:
      cpus: '1'
      memory: 2G
```

Dann neu starten:
```bash
docker compose up -d --build
```

---

### Error: `poetry.lock not found` oder `pyproject.toml changed significantly`

**Problem:** Dependencies sind nicht synchron.

**Lösung:**
```bash
# poetry.lock neu generieren
docker compose run --rm sparse-reranker poetry lock
docker compose up -d --build
```

---

### Error: `HUGGINGFACE_API_KEY not set`

**Problem:** `.env` Datei fehlt oder ist leer.

**Lösung:**
```bash
# Check ob .env existiert
cat .env

# Falls leer oder falsch:
cp .env.example .env
nano .env  # API Key eintragen
docker compose restart
```

---

### Models laden extrem langsam

**Problem:** SPLADE Model (~500MB) wird bei jedem Build neu heruntergeladen.

**Lösung:** Models Volume ist korrekt gemountet in `docker-compose.yml`:
```yaml
volumes:
  - ./models:/models  # Models werden hier persistent gespeichert
```

Nach erstem Download werden Models wiederverwendet.

---

### Service startet nicht

**Debug-Schritte:**
```bash
# 1. Logs anschauen
docker compose logs sparse-reranker

# 2. Container Status prüfen
docker ps -a

# 3. Images neu bauen (clean)
docker compose down
docker system prune -a
docker compose up -d --build

# 4. Health Check
curl http://localhost:8000/health
```

---

## 🏗️ Architektur

```
┌─────────────────────────────────────────────┐
│           Client (API Requests)             │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│      FastAPI Server (Port 8000)             │
├─────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │  SPLADE  │  │FlashRank │  │    PDF    │ │
│  │  Sparse  │  │ Reranker │  │Processing │ │
│  │  Vector  │  │          │  │           │ │
│  └──────────┘  └──────────┘  └───────────┘ │
└─────────────────────────────────────────────┘
         │                │              │
         ▼                ▼              ▼
    Qdrant          Vector DB       Document
   (extern)        (extern)         Store
```

**Integration mit bestehenden Services:**
- **Qdrant**: Hybrid Search (Dense + Sparse Vectors)
- **n8n**: Workflow-Automatisierung für PDF Processing
- **Open WebUI**: Chat-Interface mit RAG-Integration
- **Langfuse**: Monitoring und Kosten-Tracking

---

## 📚 Weitere Ressourcen

- **SPLADE v3 Model**: https://huggingface.co/naver/splade-v3-lexical
- **FlashRank**: https://github.com/PrithivirajDamodaran/FlashRank
- **Qdrant Hybrid Search**: https://qdrant.tech/articles/sparse-vectors/
- **PyMuPDF**: https://pymupdf.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/

---

## 🤝 Support & Contributing

### Support
- **GitHub Issues**: https://github.com/Sentrovo/CorporateGPT-Sparse-Reranker/issues
- **Email**: info@sentrovo.de

### Contributing
Pull Requests sind willkommen! Für größere Änderungen bitte zuerst ein Issue öffnen.

---

## 📝 Lizenz

MIT License - siehe LICENSE file

---

## 🏷️ Version

**Current Version:** 5.0.0-flashrank-integration

**Changelog:**
- v5.0.0: FlashRank Reranking Integration
- v4.0.0: Batch PDF Processing
- v3.0.0: URL-basierte PDF Verarbeitung
- v2.0.0: SPLADE v3 Upgrade
- v1.0.0: Initial Release
