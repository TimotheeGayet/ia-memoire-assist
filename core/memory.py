# === core/memory.py ===
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
MEMORIES_FILE = "memories.json"

memory_store = []
index = None

def load_memories():
    global memory_store, index
    if os.path.exists(MEMORIES_FILE):
        with open(MEMORIES_FILE, 'r', encoding='utf-8') as f:
            memory_store = json.load(f)
    if memory_store:
        embeddings = MODEL.encode(memory_store)
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(np.array(embeddings))
    else:
        dummy_embedding = MODEL.encode([""])
        index = faiss.IndexFlatL2(dummy_embedding[0].shape[0])

def save_memories():
    os.makedirs(os.path.dirname(MEMORIES_FILE) or '.', exist_ok=True)
    with open(MEMORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_store, f, ensure_ascii=False, indent=2)

def add_memory(note):
    global memory_store, index
    if note not in memory_store:
        memory_store.append(note)
        save_memories()
        vec = MODEL.encode([note])
        index.add(np.array(vec))

def retrieve_memory(query, k=3):
    if not memory_store:
        return []
    query_vec = MODEL.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [memory_store[i] for i in indices[0]]
