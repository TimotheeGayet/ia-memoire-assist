# === main.py ===
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
from core.audio import vad_record_and_transcribe, text_to_speech
from core.llm import ask_question, filter_messages
from core.memory import load_memories, add_memory, add_to_working_memory, get_working_memory
import json

async def main():
    print("=== Système de Mémoire IA ===")
    
    await text_to_speech("Bonjour ! Je suis Anna, votre assistant IA avec mémoire.")

    load_memories()
    last_user_input = None

    try:
        while True:
            print("\nAppuyez sur 1 pour commencer à parler (ou 2 pour quitter)...")
            user_choice = input()
            
            if user_choice == "2":
                print("La discussion est en train de s'enregistrer dans la mémoire...")
                if last_user_input != None:
                    summary = filter_messages(str(get_working_memory()))
                    add_memory(summary)
                print("\nAu revoir !")
                break
            elif user_choice == "1":
                user_input = vad_record_and_transcribe()
                last_user_input = user_input
                print(f"\nVous avez dit: {user_input}")
                add_to_working_memory("user", user_input)
                response = ask_question(user_input)
                print(f"\nJarvis: {response}")
                add_to_working_memory("assistant", response)
                print("\nLecture de la réponse...")
                await text_to_speech(response)
            else:
                print("Choix invalide. Veuillez réessayer.")
    except KeyboardInterrupt:
        print("\nAu revoir !")
        if last_user_input:
            summary = filter_messages(str(get_working_memory()))
            add_memory(summary)
        print(json.dumps(get_working_memory(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
