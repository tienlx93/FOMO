import os, uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI

@dataclass
class PineconeConfig:
    index_name: str = "fomo-db"
    dimension: int = 1536
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"

class PineconeVectorStore:
    """Pinecone vector store with Azure OpenAI embeddings."""
    
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
            print(f"✅ Created new Pinecone index: {cfg.index_name} with dimension {cfg.dimension}")
        else:
            # Check if existing index has matching dimension
            existing_indexes = self.pc.list_indexes()
            for idx_info in existing_indexes:
                if idx_info.name == cfg.index_name:
                    if hasattr(idx_info, 'dimension') and idx_info.dimension != cfg.dimension:
                        print(f"⚠️ WARNING: Existing index '{cfg.index_name}' has dimension {idx_info.dimension}, but config expects {cfg.dimension}")
                        print(f"⚠️ This may cause errors. Consider creating a new index or updating dimension config.")
        
        self.index = self.pc.Index(cfg.index_name)
        
        # Use Azure OpenAI for embeddings
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
        
        if not (azure_api_key and azure_endpoint and azure_embed_deployment):
            raise RuntimeError("Missing Azure OpenAI configuration. Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_EMBED_DEPLOYMENT")
        
        self.client = AzureOpenAI(
            api_key=azure_api_key,
            api_version="2024-07-01-preview",
            azure_endpoint=azure_endpoint
        )
        
        # Test embedding deployment
        test_response = self.client.embeddings.create(
            input="test",
            model=azure_embed_deployment
        )
        
        self.embed_model = azure_embed_deployment
        self.embed_fn = self._azure_embed
        # Update dimension based on embedding size
        self.cfg.dimension = len(test_response.data[0].embedding)
        print(f"✅ Using Azure OpenAI embeddings: {azure_embed_deployment} (dimension: {self.cfg.dimension})")
    
    def _azure_embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.embed_model)
        return [d.embedding for d in resp.data]
    
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
