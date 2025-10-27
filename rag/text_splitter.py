from typing import List, Dict
def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Dict[str, str]]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append({"content": " ".join(words[start:end])})
        start = end - chunk_overlap if end - chunk_overlap > start else end
    return chunks
