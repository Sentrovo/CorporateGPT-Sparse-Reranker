# === ENHANCED PDF SERVICE MIT URL-DOWNLOAD SUPPORT ===
# Server downloadet PDFs selbst und verarbeitet sie

# === Standardbibliothek ===
import os
import io
import re
import json
import base64
import tempfile
import logging
import requests
from typing import List, Dict, Optional
from urllib.parse import urlparse

# === Drittanbieter-Module ===
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# === FLASK APP SETUP ===
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === SPLADE MODEL INITIALIZATION ===
splade_model = None
splade_tokenizer = None

def initialize_splade():
    """Initialize SPLADE model and tokenizer"""
    global splade_model, splade_tokenizer

    model_id = "naver/splade-v3-lexical"
    hf_token = os.getenv('HUGGINGFACE_API_KEY')

    if not hf_token:
        raise RuntimeError("HUGGINGFACE_API_KEY not found in environment variables")

    logger.info(f"Loading SPLADE model: {model_id}")

    splade_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    splade_model = AutoModelForMaskedLM.from_pretrained(model_id, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splade_model.to(device)
    splade_model.eval()

    logger.info(f"SPLADE model loaded successfully on {device}")

def compute_sparse_vector(text: str) -> Dict:
    """Compute sparse vector using SPLADE v3"""
    tokens = splade_tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    )

    device = next(splade_model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = splade_model(**tokens)

    logits = output.logits
    relu_log = torch.log(1 + torch.relu(logits))
    max_val, _ = torch.max(relu_log, dim=1)
    vec = max_val.squeeze()

    nonzero_indices = vec.nonzero().squeeze()

    if nonzero_indices.dim() == 0:
        nonzero_indices = [nonzero_indices.item()]
        nonzero_values = [vec[nonzero_indices[0]].item()]
    else:
        nonzero_indices = nonzero_indices.tolist()
        nonzero_values = [vec[idx].item() for idx in nonzero_indices]

    return {
        "indices": nonzero_indices,
        "values": nonzero_values
    }





# === ERROR HANDLERS ===
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large",
        "max_size_mb": 100,
        "message": "Please reduce file size or use URL-based processing"
    }), 413

# === NEW URL-BASED PDF PROCESSING ===

@app.route('/pdf/process-url', methods=['POST'])
def process_pdf_from_url():
    """Download PDF from URL and process it"""
    try:
        data = request.json

        # Get parameters
        pdf_url = data.get('url')
        chunk_size = data.get('chunk_size', 1000)
        chunk_overlap = data.get('chunk_overlap', 200)
        file_name = data.get('file_name', 'document.pdf')

        if not pdf_url:
            return jsonify({"success": False, "error": "No URL provided"}), 400

        logger.info(f"Processing PDF from URL: {file_name}")

        # Download PDF from URL
        try:
            response = requests.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not file_name.lower().endswith('.pdf'):
                return jsonify({
                    "success": False, 
                    "error": f"Invalid content type: {content_type}. Expected PDF."
                }), 400

            pdf_content = response.content

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF: {str(e)}")
            return jsonify({
                "success": False, 
                "error": f"Failed to download PDF: {str(e)}"
            }), 400

        # Process the downloaded PDF
        result = process_pdf_content(pdf_content, chunk_size, chunk_overlap, file_name)

        # Add URL info to result
        result["source_url"] = pdf_url
        result["download_size_bytes"] = len(pdf_content)
        result["download_size_mb"] = round(len(pdf_content) / 1024 / 1024, 2)

        return jsonify(result)

    except Exception as e:
        logger.error(f"URL processing failed: {str(e)}")
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

@app.route('/pdf/process-urls-batch', methods=['POST'])
def process_pdfs_from_urls_batch():
    """Process multiple PDFs from URLs with parallel downloads"""
    try:
        data = request.json
        pdf_requests = data.get('files', [])

        if not pdf_requests or not isinstance(pdf_requests, list):
            return jsonify({"success": False, "error": "No files array provided"}), 400

        logger.info(f"Starting parallel processing of {len(pdf_requests)} PDFs")

        def process_single_pdf(file_request, index):
            """Process a single PDF - for threading"""
            url = file_request.get('url')
            file_name = file_request.get('file_name', f'document_{index}.pdf')
            chunk_size = file_request.get('chunk_size', 1000)
            chunk_overlap = file_request.get('chunk_overlap', 200)

            if not url:
                return {
                    "success": False,
                    "file_name": file_name,
                    "error": "No URL provided",
                    "batch_index": index
                }

            try:
                logger.info(f"Processing {index+1}/{len(pdf_requests)}: {file_name}")

                # Download PDF
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                pdf_content = response.content

                # Process PDF
                result = process_pdf_content(pdf_content, chunk_size, chunk_overlap, file_name)
                result["source_url"] = url
                result["download_size_mb"] = round(len(pdf_content) / 1024 / 1024, 2)
                result["batch_index"] = index

                return result

            except Exception as e:
                logger.error(f"Failed to process {file_name}: {str(e)}")
                return {
                    "success": False,
                    "file_name": file_name,
                    "error": str(e),
                    "batch_index": index
                }

        # Process PDFs in parallel using ThreadPool
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_single_pdf, pdf_request, i) 
                for i, pdf_request in enumerate(pdf_requests)
            ]
            results = [future.result() for future in futures]

        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful

        logger.info(f"Parallel processing complete: {successful} successful, {failed} failed")

        return jsonify({
            "success": True,
            "total_files": len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        })

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return jsonify({"success": False, "error": f"Batch processing failed: {str(e)}"}), 500

def process_pdf_content(pdf_content: bytes, chunk_size: int, chunk_overlap: int, file_name: str) -> Dict:
    """Core PDF processing function"""

    # Step 1: Extract pages with exact positions
    doc = fitz.open("pdf", pdf_content)
    pages_data = []
    total_char_position = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()

        # Clean text but preserve structure
        cleaned_text = re.sub(r'\n{3,}', '\n\n', page_text).strip()

        char_start = total_char_position
        char_end = total_char_position + len(cleaned_text)

        page_data = {
            "page_number": page_num + 1,
            "text": cleaned_text,
            "char_start": char_start,
            "char_end": char_end,
            "word_count": len(cleaned_text.split()),
            "char_count": len(cleaned_text)
        }

        pages_data.append(page_data)
        total_char_position = char_end + 2

    doc.close()

    # Step 2: Build full text and exact page boundaries
    full_text = ""
    page_boundaries = []

    for page in pages_data:
        page_start = len(full_text)
        full_text += page["text"] + "\n\n"
        page_end = len(full_text) - 2
        page_boundaries.append((page_start, page_end, page["page_number"]))

    # Step 3: Create chunks with exact position-based page mapping
    chunks_with_pages = create_intelligent_chunks_with_positions(
        full_text.strip(), chunk_size, chunk_overlap, page_boundaries
    )

    # Debug logging
    logger.info(f"Successfully processed PDF: {file_name} - {len(pages_data)} pages, {len(chunks_with_pages)} chunks")

    return {
        "success": True,
        "file_name": file_name,
        "extraction": {
            "total_pages": len(pages_data),
            "total_chars": len(full_text),
            "total_words": len(full_text.split())
        },
        "chunking": {
            "total_chunks": len(chunks_with_pages),
            "chunks": chunks_with_pages
        },
        "summary": {
            "total_pages": len(pages_data),
            "total_chunks": len(chunks_with_pages),
            "avg_chunks_per_page": len(chunks_with_pages) / len(pages_data) if pages_data else 0
        }
    }

# === EXISTING ENDPOINTS ===

@app.route('/pdf/extract-and-chunk', methods=['POST'])
def extract_and_chunk():
    """Legacy endpoint for file uploads"""
    try:
        chunk_size = int(request.form.get('chunk_size', 1000))
        chunk_overlap = int(request.form.get('chunk_overlap', 200))

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        pdf_content = file.read()
        result = process_pdf_content(pdf_content, chunk_size, chunk_overlap, file.filename)

        return jsonify(result)

    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

@app.route('/sparse/batch', methods=['POST'])
def compute_sparse_batch():
    """Sparse vector endpoint - parallel processing with threading"""
    try:
        data = request.json
        texts = data.get('texts', [])
        max_length = data.get('max_length', 512)

        if not texts or not isinstance(texts, list):
            return jsonify({"success": False, "error": "No texts array provided"}), 400

        logger.info(f"Processing {len(texts)} texts for sparse vectors using threading")

        def process_single_text(text, index):
            """Process single text for threading"""
            if len(text) > max_length * 4:
                text = text[:max_length * 4]

            result = compute_sparse_vector(text)

            if (index + 1) % 10 == 0:
                logger.info(f"Processed {index + 1}/{len(texts)} sparse vectors")

            return result

        # Process texts in parallel using ThreadPool
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(process_single_text, text, i) 
                for i, text in enumerate(texts)
            ]
            results = [future.result() for future in futures]

        return jsonify({
            "success": True,
            "count": len(texts),
            "results": results,
            "model_used": "naver/splade-v3-lexical"
        })

    except Exception as e:
        logger.error(f"Sparse vector batch computation failed: {str(e)}")
        return jsonify({"success": False, "error": f"Sparse computation failed: {str(e)}"}), 500

@app.route('/sparse', methods=['POST'])
def compute_sparse_single():
    """Sparse vector endpoint - single text processing with SPLADE"""
    try:
        if request.is_json:
            data = request.json
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        result = compute_sparse_vector(text)
        result["model_used"] = "naver/splade-v3-lexical"

        return jsonify(result)

    except Exception as e:
        logger.error(f"Sparse vector computation failed: {str(e)}")
        return jsonify({"success": False, "error": f"Sparse computation failed: {str(e)}"}), 500

# === HELPER FUNCTIONS ===

def create_intelligent_chunks_with_positions(text: str, chunk_size: int, overlap: int, page_boundaries: List) -> List[Dict]:
    """Create chunks with exact position tracking and accurate page mapping"""
    chunks = []
    current_pos = 0
    chunk_index = 0

    while current_pos < len(text):
        chunk_end = min(current_pos + chunk_size, len(text))

        if chunk_end < len(text):
            search_start = max(chunk_end - 100, current_pos)
            search_end = min(chunk_end + 50, len(text))
            search_text = text[search_start:search_end]

            sentence_patterns = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1

            for pattern in sentence_patterns:
                pos = search_text.rfind(pattern)
                if pos > 50:
                    break_pos = search_start + pos + len(pattern)
                    if break_pos > best_break:
                        best_break = break_pos

            if best_break > current_pos:
                chunk_end = best_break

        chunk_text = text[current_pos:chunk_end].strip()

        if len(chunk_text) > 50:
            overlapping_pages = find_overlapping_pages_exact(current_pos, chunk_end, page_boundaries)
            page_coverage = calculate_page_coverage_exact(current_pos, chunk_end, overlapping_pages, page_boundaries)
            primary_page = max(page_coverage.items(), key=lambda x: x[1])[0] if page_coverage else 1

            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_start": current_pos,
                "char_end": chunk_end,
                "primary_page": primary_page,
                "page_numbers": list(page_coverage.keys()),
                "page_coverage": page_coverage
            })

            chunk_index += 1

        next_pos = max(current_pos + chunk_size - overlap, chunk_end - overlap)

        if next_pos <= current_pos:
            next_pos = current_pos + 1

        current_pos = next_pos

        if current_pos >= len(text):
            break

    return chunks

def find_overlapping_pages_exact(chunk_start: int, chunk_end: int, page_boundaries: List) -> List[int]:
    """Find pages that overlap with chunk using exact positions"""
    overlapping_pages = []

    for page_start, page_end, page_num in page_boundaries:
        overlap_start = max(chunk_start, page_start)
        overlap_end = min(chunk_end, page_end)

        if overlap_start < overlap_end:
            overlapping_pages.append(page_num)

    return overlapping_pages

def calculate_page_coverage_exact(chunk_start: int, chunk_end: int, overlapping_pages: List[int], page_boundaries: List) -> Dict[int, float]:
    """Calculate exact coverage percentages"""
    coverage = {}
    chunk_length = chunk_end - chunk_start

    if chunk_length <= 0:
        return coverage

    for page_num in overlapping_pages:
        page_start, page_end, _ = next(
            (start, end, num) for start, end, num in page_boundaries if num == page_num
        )

        overlap_start = max(chunk_start, page_start)
        overlap_end = min(chunk_end, page_end)
        overlap_length = max(0, overlap_end - overlap_start)

        coverage[page_num] = overlap_length / chunk_length

    return coverage

# === HEALTH CHECK ===

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for all services"""
    model_status = "loaded" if splade_model else "not_loaded"
    device = "unknown"

    if splade_model:
        try:
            device = str(next(splade_model.parameters()).device)
        except:
            device = "unknown"

    return jsonify({
        "status": "healthy",
        "services": ["splade_sparse_vectors", "pdf_processing", "url_processing"],
        "splade_model": {
            "status": model_status,
            "model_id": "naver/splade-v3-lexical",
            "device": device
        },
        "endpoints": {
            "sparse": ["/sparse/batch", "/sparse"],
            "pdf": ["/pdf/extract-and-chunk"],
            "url_processing": ["/pdf/process-url", "/pdf/process-urls-batch"],
            "system": ["/health"]
        },
        "version": "4.0.0-splade-integration"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        "message": "Sparse Vector + PDF Processing Service (URL Download Support)",
        "services": ["sparse_vectors", "pdf_processing", "url_processing"],
        "health_check": "/health",
        "version": "3.0.0-url-support"
    })

# === ERROR HANDLERS ===

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found", 
        "available_endpoints": ["/health", "/sparse/batch", "/sparse", "/pdf/extract-and-chunk", "/pdf/process-url", "/pdf/process-urls-batch"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error", 
        "message": str(error)
    }), 500

# === MAIN ===

# Initialize SPLADE model at startup
try:
    initialize_splade()
    logger.info("SPLADE model initialization complete")
except Exception as e:
    logger.error(f"Failed to initialize SPLADE model: {str(e)}")
    raise

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)