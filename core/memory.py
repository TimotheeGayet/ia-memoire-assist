# === core/memory.py ===
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import datetime

MODEL = SentenceTransformer('all-MiniLM-L6-v2')
MEMORIES_FILE = "memories.json"

# === Mémoire de long terme ===
memory_store = []
index = None

def load_memories():
    global memory_store, index
    if os.path.exists(MEMORIES_FILE):
        with open(MEMORIES_FILE, 'r', encoding='utf-8') as f:
            memory_store = json.load(f)
    if memory_store:
        memory_texts = [m["content"] for m in memory_store]
        embeddings = MODEL.encode(memory_texts)
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(np.array(embeddings))
    else:
        dummy_embedding = MODEL.encode([""])
        index = faiss.IndexFlatL2(dummy_embedding[0].shape[0])

def save_memories():
    os.makedirs(os.path.dirname(MEMORIES_FILE) or '.', exist_ok=True)
    with open(MEMORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_store, f, ensure_ascii=False, indent=2, default=str)

def add_memory(memory):
    global memory_store, index
    
    # Si memory est une liste, on traite chaque élément
    memories = memory if isinstance(memory, list) else [memory]
    
    for mem in memories:
        # Validation du format
        required_fields = ["type", "content", "timestamp", "tags", "importance"]
        valid_types = ["preference", "fact", "event", "none"]
        
        # Vérification des champs requis
        for field in required_fields:
            if field not in mem:
                raise ValueError(f"Champ requis manquant : {field}")
        
        # Validation du type
        if mem["type"] not in valid_types:
            raise ValueError(f"Type invalide : {mem['type']}. Doit être l'un des suivants : {valid_types}")
        
        # Validation de l'importance
        if not isinstance(mem["importance"], (int, float)) or mem["importance"] < 0 or mem["importance"] > 5:
            raise ValueError("L'importance doit être un nombre entre 0 et 5")
        
        # Normalisation des tags
        if not isinstance(mem["tags"], list):
            mem["tags"] = []
        
        # Normalisation du timestamp
        if not isinstance(mem["timestamp"], str):
            mem["timestamp"] = datetime.datetime.now().isoformat()
        
        # Création de l'objet mémoire normalisé
        normalized_memory = {
            "type": mem["type"],
            "content": str(mem["content"]),  # Conversion en string pour éviter les problèmes
            "timestamp": mem["timestamp"],
            "tags": [str(tag) for tag in mem["tags"]],  # Conversion des tags en strings
            "importance": int(mem["importance"])  # Conversion en entier
        }
        
        if normalized_memory["type"] != "none":
            memory_store.append(normalized_memory)
            vec = MODEL.encode([normalized_memory["content"]])
            index.add(np.array(vec))
    
    if any(m["type"] != "none" for m in memories):
        save_memories()

def retrieve_memory(query, k=5):
    if not memory_store:
        return []
    query_vec = MODEL.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [str(memory_store[i]) for i in indices[0]]


# === Mémoire de travail ===
working_memory = []

def add_to_working_memory(role, utterance):
    """Ajoute une phrase temporairement à la mémoire de travail."""
    working_memory.append({"role": role, "content": utterance})

def get_working_memory():
    """Retourne la mémoire de travail actuelle sous forme de json utilisable par l'assistant."""
    return working_memory