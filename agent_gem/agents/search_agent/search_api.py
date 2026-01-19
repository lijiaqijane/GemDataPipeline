import argparse
import json
import os
from typing import List

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Text-to-Text Search API")

# Global variables, will be initialized at startup
model = None
index = None
doc_map = None


# --- 2. Define request format ---
class SearchQuery(BaseModel):
    text: str  # Input search text
    top_k: int = 5  # Return top K results


# --- 3. Search logic ---
@app.post("/search")
async def search_by_text(query: SearchQuery):
    try:
        # A. Convert text to vector (Sentence-Transformers)
        # normalize_embeddings=True works better with IndexFlatIP
        query_vector = model.encode([query.text])

        # B. Faiss search
        D, I = index.search(query_vector, query.top_k)

        # C. Map results
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1:
                results.append(
                    {
                        "id": int(idx),
                        "content": doc_map.get(str(idx), "Mapping Not Found"),
                        "distance": float(dist),
                    }
                )

        return {"query": query.text, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")


def initialize_resources(model_path: str, index_path: str, mapping_path: str):
    """Initialize model, index and mapping"""
    global model, index, doc_map

    print(f"Loading Embedding Model from {model_path}...")
    model = SentenceTransformer(model_path)

    print(f"Loading Faiss Index from {index_path}...")
    index = faiss.read_index(index_path)
    num_gpus = faiss.get_num_gpus()
    if num_gpus > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        print(f"FAISS index moved to {num_gpus} GPU(s)")
    else:
        print("faiss-gpu not available, using CPU")

    print(f"Loading Text Mapping from {mapping_path}...")
    with open(mapping_path, "r", encoding="utf-8") as f:
        doc_map = json.load(f)

    print("All resources loaded successfully!")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Text-to-Text Search API")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to embedding model directory (can also be set via environment variable SEARCH_MODEL_PATH)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Path to FAISS index file (can also be set via environment variable SEARCH_INDEX_PATH)",
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default=None,
        help="Path to text mapping JSON file (can also be set via environment variable SEARCH_MAPPING_PATH)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if not all([args.model_path, args.index_path, args.mapping_path]):
        raise ValueError("All model_path, index_path, and mapping_path must be specified")

    initialize_resources(args.model_path, args.index_path, args.mapping_path)

    uvicorn.run(app, host=args.host, port=args.port)
