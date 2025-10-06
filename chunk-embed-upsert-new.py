# ingest_qdrant_gemini.py
# .txt -> chunk -> Gemini embeddings (gemini-embedding-001) -> Qdrant (batched, idempotent, connection-robust)

import os, sys, uuid, time, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import google.generativeai as genai

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff,
    PayloadSchemaType, Filter, FieldCondition, MatchValue
)

# ──────────────────────────────────────────────────────────────────────────────
# ENV
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or "", override=False)

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
QDRANT_URL       = os.getenv("QDRANT_URL", "").strip()  # e.g., https://<cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_COLLECTION= os.getenv("QDRANT_COLLECTION", "sec-filings").strip()
SOURCE_DIR       = os.getenv("SOURCE_DIR", "./data").strip()

# Gemini embedding model + dims
EMBED_MODEL      = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip()  # or "text-embedding-004"
MODEL_DIMS       = {
    "gemini-embedding-001": 768,
    "text-embedding-004": 768,
}
VECTOR_SIZE      = int(os.getenv("VECTOR_SIZE", str(MODEL_DIMS.get(EMBED_MODEL, 768))))
DISTANCE         = os.getenv("DISTANCE", "COSINE").strip().upper()
RECREATE         = os.getenv("QDRANT_RECREATE", "0").lower() in {"1","true","yes"}

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBED_BATCH      = int(os.getenv("EMBED_BATCH", "64"))
UPLOAD_BATCH     = int(os.getenv("UPSERT_BATCH", "256"))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "5"))
REQUEST_TIMEOUT  = float(os.getenv("QDRANT_TIMEOUT", "180"))

# Optional proper gRPC wiring (only if you want gRPC)
QDRANT_GRPC_HOST = os.getenv("QDRANT_GRPC_HOST", "").strip()  # e.g., "<cluster>.cloud.qdrant.io"
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_REST_PORT = int(os.getenv("QDRANT_REST_PORT", "6333"))
QDRANT_HTTPS     = os.getenv("QDRANT_HTTPS", "1").lower() in {"1","true","yes"}

# Idempotency: skip entire file if any chunk for its absolute path exists
SKIP_BY_FILE     = True

# ──────────────────────────────────────────────────────────────────────────────
# SANITY
# ──────────────────────────────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    print("[FATAL] GEMINI_API_KEY missing", file=sys.stderr); sys.exit(1)

exp = MODEL_DIMS.get(EMBED_MODEL)
if exp and exp != VECTOR_SIZE:
    print(f"[FATAL] VECTOR_SIZE={VECTOR_SIZE} != {EMBED_MODEL} dims ({exp})", file=sys.stderr); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)

def _build_qdrant_client() -> QdrantClient:
    """
    Robust connection builder:
    1) If QDRANT_GRPC_HOST set -> dual-port (REST 6333 + gRPC 6334)
    2) Else, if QDRANT_URL set:
        a) Try host/port extracted from URL with https=True (most reliable on Cloud)
        b) Fall back to url=... (REST) prefer_grpc=False
    """
    # 1) Proper gRPC mode
    if QDRANT_GRPC_HOST:
        return QdrantClient(
            host=QDRANT_GRPC_HOST,
            port=QDRANT_REST_PORT,      # REST
            grpc_port=QDRANT_GRPC_PORT, # gRPC
            https=QDRANT_HTTPS,
            api_key=QDRANT_API_KEY,
            timeout=REQUEST_TIMEOUT,
            prefer_grpc=True,
            check_compatibility=False,
        )

    if not QDRANT_URL:
        print("[FATAL] Provide QDRANT_URL or QDRANT_GRPC_HOST", file=sys.stderr); sys.exit(1)

    # 2a) Derive host/port from URL and use https=True
    parsed = urlparse(QDRANT_URL)
    host = parsed.hostname
    port = parsed.port or 6333  # default REST
    if host:
        try:
            client = QdrantClient(
                host=host,
                port=port,
                https=parsed.scheme == "https" or QDRANT_HTTPS,
                api_key=QDRANT_API_KEY,
                timeout=REQUEST_TIMEOUT,
                prefer_grpc=False,      # REST
                check_compatibility=False,
            )
            client.get_collections()
            return client
        except Exception:
            pass  # try 2b

    # 2b) Fall back to the raw URL form
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=REQUEST_TIMEOUT,
        prefer_grpc=False,
        check_compatibility=False,
    )
    client.get_collections()
    return client

qd = _build_qdrant_client()

# ──────────────────────────────────────────────────────────────────────────────
# Qdrant collection management (hardened; no collection_exists())
# ──────────────────────────────────────────────────────────────────────────────
def _distance_from_str(s: str) -> Distance:
    return {"COSINE": Distance.COSINE, "DOT": Distance.DOT, "EUCLID": Distance.EUCLID}.get(s.upper(), Distance.COSINE)

def _exists_via_get(name: str) -> Tuple[bool, Optional[int], Optional[str]]:
    """Probe existence using GET /collections/{name}. Returns: (exists, size, distance_str)"""
    try:
        info = qd.get_collection(name)
        cfg = info.config.params.vectors
        cur_size = getattr(cfg, "size", None)
        cur_dist = str(getattr(cfg, "distance", "")).split(".")[-1].upper()
        return True, cur_size, cur_dist
    except UnexpectedResponse as e:
        status = getattr(e, "status", None) or getattr(e, "status_code", None)
        if status == 404:
            return False, None, None
        raise
    except Exception:
        raise

def ensure_collection():
    dist = _distance_from_str(DISTANCE)

    exists, cur_size, cur_dist = _exists_via_get(QDRANT_COLLECTION)

    if exists:
        mismatch = (cur_size != VECTOR_SIZE) or (cur_dist != DISTANCE)
        if mismatch:
            if not RECREATE:
                raise ValueError(
                    f"Vector params mismatch in '{QDRANT_COLLECTION}': "
                    f"have size={cur_size},dist={cur_dist} but need size={VECTOR_SIZE},dist={DISTANCE}. "
                    f"Set QDRANT_RECREATE=1 to rebuild."
                )
            qd.delete_collection(QDRANT_COLLECTION)
            exists = False

    if not exists or RECREATE:
        qd.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=dist),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20_000),
        )

    # payload index for fast skip-by-file (idempotent)
    try:
        qd.create_payload_index(QDRANT_COLLECTION, field_name="source_path", field_schema=PayloadSchemaType.KEYWORD)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(txt: str, size: int, overlap: int) -> List[str]:
    if size <= 0 or len(txt) <= size:
        return [txt]
    out, i, step = [], 0, max(1, size - overlap)
    while i < len(txt):
        out.append(txt[i:i+size]); i += step
    return out

def parse_meta(p: Path) -> Dict[str, Any]:
    parts = p.stem.split("__")
    meta = {"source_path": str(p.resolve()), "filename": p.name}
    if len(parts) >= 5:
        meta.update({
            "ticker":  parts[0],
            "form":    parts[1],
            "period":  parts[2],
            "sector":  parts[3],
            "company": parts[4].replace("-", " ")
        })
    return meta

def _sleep_backoff(attempt: int):
    base = min(30.0, (1.5 ** attempt))
    time.sleep(base + random.uniform(0.0, 0.25 * base))

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Gemini batch embeddings using batch_embed_contents.
    Uses task_type='RETRIEVAL_DOCUMENT' for document chunks.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Build batch requests
            reqs = [{"model": EMBED_MODEL, "content": t, "task_type": "RETRIEVAL_DOCUMENT"} for t in texts]
            resp = genai.batch_embed_contents(requests=reqs)
            # Python SDK returns .embeddings list with each having .values
            # (Handle both object and dict forms for safety)
            out = []
            for e in getattr(resp, "embeddings", resp.get("embeddings", [])):
                vals = getattr(e, "values", e.get("values"))
                out.append(list(vals))
            return out
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)

def file_already_ingested(source_path_abs: str) -> bool:
    if not SKIP_BY_FILE:
        return False
    f = Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=source_path_abs))])
    try:
        cnt = qd.count(collection_name=QDRANT_COLLECTION, count_filter=f, exact=True).count
        return cnt > 0
    except Exception:
        # If counting fails, do not skip to avoid false negatives
        return False

def upload_points(vectors: List[List[float]], payloads: List[Dict[str, Any]]):
    assert len(vectors) == len(payloads)
    def gen():
        for vec, pl in zip(vectors, payloads):
            yield PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pl)
    qd.upload_points(
        collection_name=QDRANT_COLLECTION,
        points=gen(),
        batch_size=UPLOAD_BATCH,                        # 64–256 recommended
        parallel=int(os.getenv("QDRANT_PARALLEL", "2")),
        max_retries=int(os.getenv("QDRANT_MAX_RETRIES", "5")),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    src = Path(SOURCE_DIR)
    files = sorted(src.rglob("*.txt"))
    if not files:
        print(f"[FATAL] No .txt files under {src.resolve()}", file=sys.stderr); sys.exit(1)

    ensure_collection()
    total_chunks = 0

    for f in tqdm(files, desc="Files", unit="file"):
        # read file (quietly skip unreadable)
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        meta_base = parse_meta(f)
        if file_already_ingested(meta_base["source_path"]):
            tqdm.write(f"skip: {f.name}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        batches = [chunks[i:i+EMBED_BATCH] for i in range(0, len(chunks), EMBED_BATCH)]
        pb = tqdm(total=len(chunks), leave=False, unit="chunk", desc=f"{f.name[:40]}")
        try:
            running_idx = 0
            for batch_texts in batches:
                vecs = embed_batch(batch_texts)
                payloads = [{**meta_base, "chunk_idx": running_idx + j, "text": t}
                            for j, t in enumerate(batch_texts)]
                upload_points(vecs, payloads)
                pb.update(len(batch_texts))
                total_chunks += len(batch_texts)
                running_idx += len(batch_texts)
        finally:
            pb.close()

    tqdm.write(f"Done. Embedded & upserted chunks: {total_chunks}")

if __name__ == "__main__":
    main()
