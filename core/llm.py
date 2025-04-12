# === core/llm.py ===
from openai import OpenAI
from core.memory import retrieve_memory, get_working_memory
import json
import datetime
import time

CLIENT = OpenAI()

def ask_question(query):
    context = "\n".join(retrieve_memory(query))
    past_messages = get_working_memory()
    response = CLIENT.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Tu es un assistant personnel nommé Anna. Écris en français, de façon naturelle, sincère et authentique, en t'adaptant au discours de ton interlocuteur."},
            {"role": "system", "content": f"Voici des souvenirs enregistrés, tu n'es pas obligé de les utiliser, ils sont là pour te donner un contexte qui peut te servir dans la discussion. Voici les souvenirs :\n{context}"},
            *past_messages,
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

def filter_messages(messages):
    max_attempts = 5
    attempt = 0
    delay_between_attempts = 1
    
    while attempt < max_attempts:
        try:
            response = CLIENT.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": """Tu es un agent de classification de souvenirs très sélectif. Ta mission est d'analyser si l'entrée contient des informations importantes qui méritent d'être mémorisées à long terme.

Règles strictes :
- Ne retiens que les informations factuelles, les préférences ou les événements significatifs
- Ignore complètement les formules de politesse, remerciements, salutations
- Ignore les questions simples ou les demandes d'information
- Condense chaque information en une phrase claire et concise
- Réponds UNIQUEMENT avec un JSON valide contenant une liste de souvenirs :
  {
    "memories": [
      {
        "type": "preference" | "fact" | "event" | "none",
        "content": "description concise de l'information",
        "timestamp": "date et heure actuelle au format ISO",
        "tags": ["liste", "de", "mots-clés"],
        "importance": 1-5
      }
    ]
  }
- Si l'entrée ne contient rien de mémorable, renvoie {"memories": [{"type": "none", "content": "N/A", "timestamp": "date actuelle", "tags": [], "importance": 0}]}
- NE JAMAIS ajouter de texte en dehors du JSON
- Les tags doivent être des mots-clés pertinents, maximum 5 tags par souvenir
- Le contenu doit être une phrase concise de maximum 100 caractères par souvenir
- Assure-toi que le JSON est valide et peut être parsé

Input à analyser : """ + messages}
                ],
                response_format={ "type": "json_object" }
            )
            
            # Tentative de parsing du JSON
            results = json.loads(response.choices[0].message.content)
            
            # Vérification de la structure du JSON
            if "memories" not in results:
                raise ValueError("Le JSON doit contenir une clé 'memories'")
            
            memories = results["memories"]
            
            # Vérification que c'est bien une liste
            if not isinstance(memories, list):
                memories = [memories]
            
            validated_results = []
            for memory in memories:
                # Vérification des champs requis
                required_fields = ["type", "content", "timestamp", "tags", "importance"]
                if not all(field in memory for field in required_fields):
                    continue
                
                # Vérification des types valides
                valid_types = ["preference", "fact", "event", "none"]
                if memory["type"] not in valid_types:
                    continue
                
                # Vérification de l'importance
                if not isinstance(memory["importance"], (int, float)) or memory["importance"] < 0 or memory["importance"] > 5:
                    continue
                
                # Vérification du contenu
                if not isinstance(memory["content"], str) or len(memory["content"]) > 100:
                    continue
                
                # Vérification des tags
                if not isinstance(memory["tags"], list):
                    memory["tags"] = []
                if len(memory["tags"]) > 5:
                    memory["tags"] = memory["tags"][:5]
                
                # Vérification du timestamp
                try:
                    datetime.datetime.fromisoformat(memory["timestamp"])
                except ValueError:
                    memory["timestamp"] = datetime.datetime.now().isoformat()
                
                validated_results.append(memory)
            
            if not validated_results:
                validated_results = [{
                    "type": "none",
                    "content": "N/A",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tags": [],
                    "importance": 0
                }]
            
            return validated_results
            
        except (json.JSONDecodeError, ValueError) as e:
            attempt += 1
            if attempt == max_attempts:
                print(f"Échec après {max_attempts} tentatives. Utilisation d'une valeur par défaut.")
                return [{
                    "type": "none",
                    "content": "N/A",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tags": [],
                    "importance": 0
                }]
            print(f"Tentative {attempt} échouée : {str(e)}. Nouvelle tentative dans {delay_between_attempts} secondes...")
            time.sleep(delay_between_attempts)
