<<<<<<< HEAD
# EDMS Middleware: Semantic Search Integration

This document outlines the architecture and implementation of the semantic search feature, which uses ChromaDB and a separate embedding microservice to provide intelligent search capabilities with a fallback to the traditional keyword-based search.

## 1. How It Works: The Architecture

The new search functionality is composed of four main parts:

- **Embedding Service**: A standalone FastAPI microservice (`app.py`). It connects to the GovAI API to use the `qwen3-embedding-8b` embedding model and exposes an API endpoint (`/embed`) that converts text into a vector (a list of numbers).

- **Vector Client (`vector_client.py`)**: A new module in the main middleware application. This client manages all communication with ChromaDB and the Embedding Service.

- **Vector Database (ChromaDB)**: We use ChromaDB in "persistent" mode. It stores a collection (`edms_documents`) on the server's local disk (in the `./chroma_db` directory). This database stores the vector for each document's abstract, mapped to its docnumber.

- **Modified Search Logic**: The main document search function (`db_connector.fetch_documents_from_oracle`) has been updated to orchestrate the new search flow.

## 2. The Search (Read) Flow

When a user performs a search in the viewer:

1. The request hits the `/api/documents` endpoint in `app.py`.
2. This calls `db_connector.fetch_documents_from_oracle` with the `search_term`.
3. **Vector Search**: The function first calls `vector_client.query_documents(search_term)`.
4. The `vector_client` sends the `search_term` to the Embedding Service to get a vector.
5. It then uses this vector to query ChromaDB for the `n_results` (e.g., 40) most semantically similar document IDs.

**Fallback Logic (Critical):**

- IF the vector search succeeds and returns a list of doc_ids, the Oracle query is modified to fetch only those specific IDs (`WHERE p.docnumber IN (...)`) and orders them by relevance.
- IF the vector search fails (e.g., the embedding service is down) OR it returns zero results, the function "falls back" to the original SQL `LIKE` query.

All other filters (tags, persons, dates) are applied on top of either the vector results or the keyword results, ensuring filtering remains consistent.

## 3. The Indexing (Write) Flow

A document's vector must be updated whenever its abstract changes. We have added "hooks" to automatically update the vector store in the following places:

- **AI Processing**: When the `/process-batch` job finishes and successfully generates a new abstract (status = 3), `update_document_processing_status` calls `vector_client.add_or_update_document()`.
- **Manual Abstract Edit**: When a user edits the abstract/date via `api_update_metadata`, the `update_document_metadata` function calls `vector_client.add_or_update_document()`.
- **VIP Tagging**: When VIPs are added/removed, `update_abstract_with_vips` calls `vector_client.add_or_update_document()`.
- **Person Tag Deletion**: When a person is removed from the abstract, `delete_tag_from_document` calls `vector_client.add_or_update_document()`.

This `add_or_update_document` function uses ChromaDB's upsert command, which efficiently creates or overwrites the vector for a given doc_id.

## 4. Setup and Installation

### Embedding Service

1. Navigate to the Embedding Service directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file (copy from `.env.example`):
   ```
   GOVAI_API_URL=https://your-govai-api-url/v1/chat/completions
GOVAI_API_KEY=your-api-key-here
   GOVAI_EMBEDDING_MODEL=qwen3-embedding-8b
   ```
6. Run the service: `python app.py` (It will run on port 5003).

### Main Middleware Application

1. Activate your main venv: `venv\Scripts\activate`
2. Install the new dependency: `pip install chromadb`
3. Add the new variables (see below) to your `.env` file.
4. Run the main application as usual: `run_api.bat` (It will run on port 5000).

### New .env Variables

Add these to your `edms-middleware-py/.env` file:

```
EMBEDDING_API_URL="http://localhost:5003"
CHROMA_DB_PATH="./chroma_db"
CHROMA_COLLECTION_NAME="edms_documents"
```

### Update .gitignore

Add `/chroma_db/` to your main `.gitignore` file to avoid committing the vector database.

## 5. Offline Deployment

For deploying to an offline server, use the provided batch scripts:

### On an Internet-Connected Machine

1. Run `download_packages.bat` to download all Python wheel files into the `packages/` folder.
2. Run `create_archive.bat` to create a timestamped ZIP archive.

### On the Offline Server

1. Extract the ZIP archive.
2. Run `install_packages.bat` to set up the virtual environment and install packages.
3. Create `.env` file with your govAI API credentials.
4. Run `run.bat` for local testing, or deploy to IIS using `web.config`.

## 6. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/embed` | Get embedding vector for text |
| GET | `/health` | Health check endpoint |
| GET | `/docs` | FastAPI interactive documentation (Swagger UI) |
| GET | `/redoc` | FastAPI alternative documentation |

### Example Request

```bash
curl -X POST http://localhost:5003/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

### Example Response

```json
{
  "text": "Your text here",
  "embedding": [0.123, -0.456, ...]
}
```

## 7. Backfilling Existing Documents

Your new search will only work on documents that are indexed. You must run the backfill script to index all existing documents in your database.

1. Ensure the new Embedding Service (port 5003) is running. (The main middleware API on port 5000 does not need to be running for the backfill).
2. Open a new terminal.
3. Activate the main middleware's virtual environment: `venv\Scripts\activate`
4. Run the backfill script:
   ```
   python backfill_vectors.py
   ```

=======
EDMS Middleware: Semantic Search Integration

This document outlines the architecture and implementation of the semantic search feature, which uses ChromaDB and a separate embedding microservice to provide intelligent search capabilities with a fallback to the traditional keyword-based search.

1. How It Works: The Architecture

The new search functionality is composed of four main parts:

Embedding Service: A new, standalone Flask microservice (embedding_service/app.py). Its only job is to load a heavy machine-learning model (sentence-transformers) and expose an API endpoint (/embed) that converts text into a vector (a list of numbers).

Vector Client (vector_client.py): A new module in the main middleware application. This client manages all communication with ChromaDB and the Embedding Service.

Vector Database (ChromaDB): We use ChromaDB in "persistent" mode. It stores a collection (edms_documents) on the server's local disk (in the ./chroma_db directory). This database stores the vector for each document's abstract, mapped to its docnumber.

Modified Search Logic: The main document search function (db_connector.fetch_documents_from_oracle) has been updated to orchestrate the new search flow.

2. The Search (Read) Flow

When a user performs a search in the viewer:

The request hits the /api/documents endpoint in app.py.

This calls db_connector.fetch_documents_from_oracle with the search_term.

Vector Search: The function first calls vector_client.query_documents(search_term).

The vector_client sends the search_term to the Embedding Service to get a vector.

It then uses this vector to query ChromaDB for the n_results (e.g., 40) most semantically similar document IDs.

Fallback Logic (Critical):

IF the vector search succeeds and returns a list of doc_ids, the Oracle query is modified to fetch only those specific IDs (WHERE p.docnumber IN (...)) and orders them by relevance.

IF the vector search fails (e.g., the embedding service is down) OR it returns zero results, the function "falls back" to the original SQL LIKE query.

All other filters (tags, persons, dates) are applied on top of either the vector results or the keyword results, ensuring filtering remains consistent.

3. The Indexing (Write) Flow

A document's vector must be updated whenever its abstract changes. We have added "hooks" to automatically update the vector store in the following places:

AI Processing: When the /process-batch job finishes and successfully generates a new abstract (status = 3), update_document_processing_status calls vector_client.add_or_update_document().

Manual Abstract Edit: When a user edits the abstract/date via api_update_metadata, the update_document_metadata function calls vector_client.add_or_update_document().

VIP Tagging: When VIPs are added/removed, update_abstract_with_vips calls vector_client.add_or_update_document().

Person Tag Deletion: When a person is removed from the abstract, delete_tag_from_document calls vector_client.add_or_update_document().

This add_or_update_document function uses ChromaDB's upsert command, which efficiently creates or overwrites the vector for a given doc_id.

4. Setup and Installation

Embedding Service:

Navigate to the new embedding_service/ directory.

Create a virtual environment: python -m venv venv

Activate it: venv\Scripts\activate

Install dependencies: pip install -r requirements.txt (This will download torch and sentence-transformers, which may take time).

Run the service: python app.py (It will run on port 5003).

Main Middleware Application:

Activate your main venv: venv\Scripts\activate

Install the new dependency: pip install chromadb

Add the new variables (see below) to your .env file.

Run the main application as usual: run_api.bat (It will run on port 5000).

New .env Variables:
Add these to your edms-middleware-py/.env file:

EMBEDDING_API_URL="http://localhost:5003"
CHROMA_DB_PATH="./chroma_db"
CHROMA_COLLECTION_NAME="edms_documents"


Update .gitignore:
Add /chroma_db/ to your main .gitignore file to avoid committing the vector database.

5. Backfilling Existing Documents

Your new search will only work on documents that are indexed. You must run the backfill script to index all existing documents in your database.

Ensure the new Embedding Service (port 5003) is running. (The main middleware API on port 5000 does not need to be running for the backfill).

Open a new terminal.

Activate the main middleware's virtual environment: venv\Scripts\activate

Run the backfill script:

python backfill_vectors.py


>>>>>>> 0334e8cddadc778e910a61669ab0c5bc489c6c40
This script will fetch documents in batches of 100 from your Oracle DB, get their embeddings via the service, and insert them into ChromaDB. This process will take a very long time given your database size, but it can be stopped and restarted (it will pick up from the last docnumber it processed, though the current simple script restarts from the beginning—it can be improved to save its progress).