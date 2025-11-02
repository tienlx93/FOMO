import os, uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except Exception:
    _FASTEMBED_AVAILABLE = False

@dataclass
class PineconeConfig:
    index_name: str = "fomo-guides"
    dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"

class PineconeVectorStore:
    """Pinecone vector store with flexible embedding backends."""
    
    def __init__(self, cfg: PineconeConfig):
        self.cfg = cfg
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise RuntimeError("Missing PINECONE_API_KEY environment variable")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        if cfg.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=cfg.index_name,
                dimension=cfg.dimension,
                metric=cfg.metric,
                spec=ServerlessSpec(
                    cloud=cfg.cloud,
                    region=cfg.region
                )
            )
        
        self.index = self.pc.Index(cfg.index_name)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        
        self.embed_fn = None
        if openai_api_key and openai_base_url:
            try:
                self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
                self.embed_model = openai_embed_model
                self.embed_fn = self._openai_embed
            except Exception:
                self.embed_fn = None
        
        if self.embed_fn is None:
            if not _FASTEMBED_AVAILABLE:
                raise RuntimeError("No embeddings available: OpenAI failed and fastembed not installed.")
            model_name = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
            self._fastembed_model = TextEmbedding(model_name=model_name)
            self.embed_fn = self._fastembed_embed
    
    def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.embed_model)
        return [d.embedding for d in resp.data]
    
    def _fastembed_embed(self, texts: List[str]) -> List[List[float]]:
        return [vec for vec in self._fastembed_model.embed(texts)]
    
    def upsert_texts(self, docs: List[Dict[str, str]], source: str = "upload"):
        vectors = []
        for d in docs:
            _id = d.get("id") or str(uuid.uuid4())
            text = d["content"]
            embedding = self.embed_fn([text])[0]
            meta = d.get("metadata", {}).copy()
            meta.setdefault("source", source)
            meta["content"] = text
            vectors.append({
                "id": _id,
                "values": embedding,
                "metadata": meta
            })
        
        if vectors:
            self.index.upsert(vectors=vectors)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        query_embedding = self.embed_fn([query])[0]
        
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        out = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            out.append({
                "id": match["id"],
                "content": metadata.get("content", ""),
                "metadata": {k: v for k, v in metadata.items() if k != "content"},
                "score": match.get("score")
            })
        return out
