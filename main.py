# === main.py ===
import asyncio
from core.audio import record_audio, transcribe_audio, text_to_speech
from core.llm import ask_question, filter_input
from core.memory import load_memories, add_memory

async def main():
    print("=== Système de Mémoire IA ===")
    await text_to_speech("Bonjour ! Je suis votre assistant personnel. Je vais mémoriser nos conversations.")
    await text_to_speech("Vous pouvez me parler normalement, et je retiendrai ce que vous me dites.")
    await text_to_speech("Pour quitter, dites 'quitter' ou 'exit'.\n")

    load_memories()

    while True:
        print("\nAppuyez sur Entrée pour commencer à parler...")
        input()
        frames, sample_rate = record_audio()
        user_input = transcribe_audio(frames, sample_rate)
        print(f"\nVous avez dit: {user_input}")

        if user_input.lower() in ['quitter', 'exit']:
            print("\nAu revoir !")
            break

        summary = filter_input(user_input)
        if summary.lower() != "n/a":
            add_memory(summary)

        response = ask_question(user_input)
        print(f"\nAssistant: {response}")
        print("\nLecture de la réponse...")
        await text_to_speech(response)

if __name__ == "__main__":
    asyncio.run(main())