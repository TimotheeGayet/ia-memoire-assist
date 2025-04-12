# 🧠 Mémoire IA – Assistant personnel cognitif inspiré des travaux sur la mémoire humaine

## Présentation

Ce projet est un prototype fonctionnel d’**assistant IA personnel à mémoire active**, construit comme **exploration pratique** du document de recherche :

> **"Human-Inspired Perspectives: A Survey on AI Long-term Memory"**  
> *arXiv:2411.00489v2 – Novembre 2024*

🎯 L'objectif est de **modéliser un système de mémoire artificielle** basé sur une hybridation inspirée du fonctionnement **humain de la mémoire** : mémoire de travail, mémoire épisodique, oubli adaptatif, condensation, et récupération sémantique contextuelle.

## ⚙️ Fonctionnalités principales

- 🎙️ **Entrée vocale** (via microphone, transcription par Whisper)
- 🧠 **Filtrage cognitif** des souvenirs : seuls les éléments pertinents sont conservés
- 🧾 **Mémorisation vectorielle** avec encodage sémantique (`SentenceTransformers + FAISS`)
- 🔍 **Récupération contextuelle (RAG)** pour assister la génération
- 🗣️ **Réponse vocale** en streaming (`gpt-4o TTS`)
- 🔁 **Mémoire persistante et évolutive**

## 🧬 Inspirations cognitives (basé sur le papier)

Le système est structuré autour de **3 couches de mémoire inspirées du modèle SALM** :

| Type de mémoire         | Fonction IA                           | Implémentation actuelle         |
|-------------------------|----------------------------------------|---------------------------------|
| 🧠 Mémoire de travail    | Contexte de la session active          | `main.py`, buffer d'entrée      |
| 🧾 Mémoire épisodique    | Souvenirs vécus / interactions         | `memory.py + FAISS + JSON`      |
| 🛠️ Mémoire procédurale   | Filtrage / structuration / condensation | `llm.py > GPT-3.5 classification` |

Objectif à terme : enrichir la mémoire avec des **liens conceptuels dynamiques**, une **pondération contextuelle adaptative** et un **oubli progressif**, comme suggéré dans l'architecture SALM du papier.

## 📁 Structure du projet

```bash
ia-memoire-assist/
├── main.py                  # Point d'entrée principal
├── core/
│   ├── audio.py             # Enregistrement, transcription, voix
│   ├── memory.py            # Gestion mémoire vectorielle
│   ├── llm.py               # Dialogue, prompts, filtres
│   └── constants.py         # Paramètres de config
├── memories.json            # Base des souvenirs
├── requirements.txt         # Dépendances Python
├── .env                     # Clé API OpenAI
└── README.md                # Documentation
```

## 🔍 Dépendances

- openai
- faiss-cpu
- pyaudio
- simpleaudio
- sentence-transformers

Utilise les modèles OpenAI gpt-3.5-turbo, gpt-4o, et whisper-1.

## 🚀 Lancer le projet

```bash
git clone https://github.com/tonprofil/ia-memoire-assist
cd ia-memoire-assist
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
touch .env  # Ajouter OPENAI_API_KEY dedans et retirer le .example
python main.py
```

## 📌 Objectifs futurs

- Mise en place d'une mémoire technique (faits, chiffres, connaissances)
- Intégration d'un réseau d'associations entre souvenirs
- Ajout d'un mécanisme d'oubli naturel (par fréquence d'accès)
- Visualisation dynamique de la mémoire (graphe, poids)
- Interface utilisateur graphique (Tauri, Streamlit, etc.)
- Mode multi-utilisateur avec sessions et personnalités

## 🧪 Pourquoi ce projet ?

Ce projet est à la fois :

- Un laboratoire cognitif pour explorer l'implémentation concrète de théories de mémoire humaine en IA
- Un prototype d'outil personnel pour la gestion d'une mémoire numérique longue durée
- Un socle pour des agents intelligents hybrides, adaptables et personnalisables

## 📄 Licence

MIT – Tu peux t'en servir, le modifier, le forker, tant que tu cites l'auteur et contribues à ton tour.

## 🙌 Contribuer

Si ce sujet t'inspire, tu peux :

- Ouvrir des issues
- Proposer des optimisations mémoire / cognition
- Suggérer des interfaces ou fonctionnalités
- Ou juste discuter de mémoire humaine, IA, et architecture cognitive :)

## 🔗 Référence

- Human-Inspired Perspectives: A Survey on AI Long-term Memory
- arXiv:2411.00489v2
- Lien à ajouter une fois le papier attaché 📎


