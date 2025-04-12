# === core/llm.py ===
from openai import OpenAI
from core.memory import retrieve_memory, add_memory

CLIENT = OpenAI()

def ask_question(query):
    memories = retrieve_memory(query)
    context = "\n".join(memories)
    prompt = (
        "Tu es un assistant personnel. Voici des souvenirs enregistrés, tu n'es pas obligé de les utiliser, ils sont là pour te donner un contexte qui peut te servir dans la discussion :\n"
        f"{context}\n\nRéponds à l'utilisateur :\n{query}"
    )
    response = CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Tu es un assistant personnel de vie."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def filter_input(user_input):
    response = CLIENT.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un agent de classification de souvenirs chargé de jauger ce qui doit être retenu et de le condenser en un souvenir. Tu dois retourner uniquement le souvenir condensé ou 'N/A' si ce n'est pas pertinent. l'input est : " + user_input}
        ]
    )
    return response.choices[0].message.content
