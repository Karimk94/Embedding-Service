import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables from a .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Embedding Service API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GovAI Embedding API Configuration ---
GOVAI_API_URL = os.getenv("GOVAI_API_URL")
GOVAI_API_KEY = os.getenv("GOVAI_API_KEY")
GOVAI_EMBEDDING_MODEL = os.getenv("GOVAI_EMBEDDING_MODEL", "qwen3-embedding-8b")

# Validate configuration
if not GOVAI_API_URL or not GOVAI_API_KEY:
    logging.critical("GOVAI_API_URL and GOVAI_API_KEY must be set in environment variables.")
    logging.critical("Please create a .env file with: GOVAI_API_URL, GOVAI_API_KEY")
else:
    logging.info(f"GovAI Embedding API configured with model: {GOVAI_EMBEDDING_MODEL}")


# --- Request Model ---
class EmbedRequest(BaseModel):
    text: str


# --- Response Model ---
class EmbedResponse(BaseModel):
    text: str
    embedding: list


class HealthResponse(BaseModel):
    status: str
    model: str = None
    reason: str = None


def get_embedding_from_govai(text: str) -> list:
    """
    Get embedding from GovAI API using OpenAI-compatible embedding endpoint.
    Returns the embedding vector or raises an exception.
    """
    # Construct embedding endpoint URL (typically /embeddings for OpenAI-compatible APIs)
    embedding_url = GOVAI_API_URL.replace("/chat/completions", "/embeddings")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GOVAI_API_KEY}",
    }
    
    payload = {
        "model": GOVAI_EMBEDDING_MODEL,
        "input": text,
    }
    
    response = requests.post(
        embedding_url,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    
    # Extract embedding from OpenAI-compatible response format
    embedding = result.get("data", [{}])[0].get("embedding", [])
    
    if not embedding:
        raise ValueError("No embedding returned from API")
    
    return embedding


@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(req: EmbedRequest):
    if not GOVAI_API_URL or not GOVAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Embedding service is not configured. Missing API credentials."
        )

    try:
        embedding = get_embedding_from_govai(req.text)
        return EmbedResponse(text=req.text, embedding=embedding)

    except requests.exceptions.RequestException as e:
        logging.error(f"API request error for text: {req.text[:50]}... Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to embedding API."
        )
    
    except (KeyError, IndexError, ValueError) as e:
        logging.error(f"API response parsing error for text: {req.text[:50]}... Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to parse embedding API response."
        )
    
    except Exception as e:
        logging.error(f"Error generating embedding for text: {req.text[:50]}... Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embedding."
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service status."""
    if not GOVAI_API_URL or not GOVAI_API_KEY:
        return HealthResponse(status="unhealthy", reason="Missing API configuration")
    
    return HealthResponse(status="healthy", model=GOVAI_EMBEDDING_MODEL)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    logging.info(f"Starting Embedding Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
