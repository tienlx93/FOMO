import os, uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions

# --- OPENAI (optional) ---
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

# --- FASTEMBED (local CPU) ---
try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except Exception:
    _FASTEMBED_AVAILABLE = False

@dataclass
class ChromaConfig:
    persist_dir: str = ".chroma"
    collection_name: str = "fomo_guides"
    metadata: Optional[dict] = None

class OpenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        if not (OPENAI_API_KEY and OPENAI_BASE_URL):
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL")
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.model = OPENAI_EMBED_MODEL

    def __call__(self, input: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=input, model=self.model)
        return [d.embedding for d in resp.data]

class FastEmbedEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Lightweight local embedding (CPU) – no API key needed.
    Default model: 'BAAI/bge-small-en-v1.5' (384 dims)
    """
    def __init__(self, model_name: str = None):
        name = model_name or os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = TextEmbedding(model_name=name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        # fastembed returns generator of vectors
        return [vec for vec in self._model.embed(input)]

class ChromaVectorStore:
    """Chroma 0.5+ with PersistentClient and flexible embedding backends."""
    def __init__(self, cfg: ChromaConfig):
        self.cfg = cfg
        os.makedirs(cfg.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=cfg.persist_dir)

        # Choose embedding backend:
        # 1) Try OpenAI embeddings if model likely accessible
        self.ef = None
        if OPENAI_API_KEY and OPENAI_BASE_URL:
            try:
                # sanity probe: many gateways reject non-allowed models → catch early
                self.ef = OpenAIEmbeddingFunction()
                # if your key is restricted, the first actual call will raise; we will handle at upsert time
            except Exception:
                self.ef = None

        # 2) Fallback to fastembed (local)
        if self.ef is None:
            if not _FASTEMBED_AVAILABLE:
                raise RuntimeError("No embeddings available: key has no embed perms and fastembed not installed.")
            self.ef = FastEmbedEmbeddingFunction()

        # Create collection (avoid empty metadata)
        if cfg.metadata:
            self.collection = self.client.get_or_create_collection(
                name=cfg.collection_name,
                metadata=cfg.metadata,
                embedding_function=self.ef,
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=cfg.collection_name,
                embedding_function=self.ef,
            )

    def upsert_texts(self, docs: List[Dict[str, str]], source: str = "upload"):
        ids, metadatas, texts = [], [], []
        for d in docs:
            _id = d.get("id") or str(uuid.uuid4())
            ids.append(_id)
            texts.append(d["content"])
            meta = d.get("metadata", {}).copy()
            meta.setdefault("source", source)
            metadatas.append(meta)
        if texts:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        res = self.collection.query(query_texts=[query], n_results=k)
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        scores = res.get("distances", [[]])[0] or []
        out = []
        for i, doc in enumerate(documents):
            out.append({
                "id": ids[i] if i < len(ids) else None,
                "content": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": scores[i] if i < len(scores) else None
            })
        return out
