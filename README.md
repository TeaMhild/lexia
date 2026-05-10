# LexIA ⚖️ — Assistant juridique RAG

[![HuggingFace Space](https://img.shields.io/badge/🤗-Demo%20Live-blue)](https://huggingface.co/spaces/Majeylent/lexia-api)
[![API Docs](https://img.shields.io/badge/API-Swagger-green)](https://majeylent-lexia-api.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Assistant juridique basé sur une architecture **RAG (Retrieval-Augmented Generation)**
sur le droit français — Code du travail + Code de la consommation.

> ⚠️ LexIA est un projet pédagogique. Les réponses ne constituent pas un conseil juridique.

---

## 🎯 Cas d'usage


Utilisateur : "Mon employeur peut-il me licencier pendant un arrêt maladie ?"

LexIA : "Selon l'article L1226-9 du Code du travail (https://www.legifrance.gouv.fr/...), au cours des périodes de suspension du contrat de travail, l'employeur ne peut rompre ce dernier que s'il justifie d'une faute grave..."

⚠️ Cette réponse est fournie à titre informatif uniquement.

---

## 🏗️ Architecture

```
Question utilisateur
│
│ Embedding (paraphrase-multilingual-mpnet-base-v2)
▼
Chroma Vector Store ──── 27 781 chunks juridiques
│
│ Similarity search (cosine, min_score=0.58)
▼
Top-5 articles pertinents (parent-child retrieval)
│
│ Prompt strict (grounding juridique)
▼
Groq Llama 3.3 70B
│
▼
Réponse sourcée + URLs Légifrance + Disclaimer
│
▼
Langfuse (monitoring — traces, latence, qualité)
```

---

## 📊 Corpus

| Code | Articles | Source |
|---|---|---|
| Code du travail | 11 496 | Dump DILA open data |
| Code de la consommation | 2 148 | Dump DILA open data |
| **Total** | **13 644** | legifrance.gouv.fr |

→ [Documentation complète du corpus](data/processed/CORPUS.md)

---

## 🚀 Demo

API déployée sur HuggingFace Spaces :

```bash
curl -X POST "https://majeylent-lexia-api.hf.space/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Mon employeur peut-il me licencier pendant un arrêt maladie ?"}'
```

Swagger UI : [https://majeylent-lexia-api.hf.space/docs](https://majeylent-lexia-api.hf.space/docs)
→ POST /api/v1/query → Try it out → Execute

Python```
import requests
response = requests.post(
    "https://majeylent-lexia-api.hf.space/api/v1/query",
    json={"question": "Quel est le délai de rétractation pour un achat en ligne ?"}
)
print(response.json()["answer"])
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| **Ingestion** | Python, xml.etree, dump DILA |
| **Chunking** | LangChain RecursiveCharacterTextSplitter |
| **Embedding** | sentence-transformers (paraphrase-multilingual-mpnet-base-v2) |
| **Vector store** | Chroma (HNSW, cosine similarity) |
| **LLM** | Groq API — Llama 3.3 70B |
| **API** | FastAPI + Uvicorn |
| **Monitoring** | Langfuse |
| **Évaluation** | RAGAS (faithfulness=0.46, context_precision=0.63) |
| **Déploiement** | HuggingFace Spaces (Docker) |

---

## 📁 Structure du projet

```
lexia/
├── ingestion/          # Phase 1 — Parsing XML LEGI + chunking
├── indexing/           # Phase 2-3 — Embedding + Chroma
├── rag/                # Phase 3 — Retriever + Prompt + Chain
├── api/                # Phase 4 — FastAPI
├── evaluation/          # Phase 5 — RAGAS (faithfulness=0.463)
├── monitoring/          # Phase 6 — Langfuse
├── notebooks/          # Exploration et démonstration
├── docs/               # Documentation architecture
└── data/processed/     # CORPUS.md + métadonnées
```

---

## ⚡ Lancer en local

```bash
# 1. Cloner
git clone https://github.com/TeaMhild/lexia.git
cd lexia

# 2. Installer
pip install -r requirements.txt
pip install -e .

# 3. Variables d'environnement
cp .env.example .env
# Remplissez GROQ_API_KEY et LANGFUSE_*

# 4. Reconstruire le corpus (voir CORPUS.md)
# ou utiliser l'index pré-généré (non versionné)

# 5. Lancer l'API
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## 📈 Évaluation RAGAS

| Métrique | Score | Interprétation |
|---|---|---|
| Faithfulness | 0.463 | Réponses ancrées dans les articles |
| Context Precision | 0.631 | 63% des chunks retrieved sont pertinents |
| Answer Relevancy | n/a | Incompatibilité RAGAS/Groq (voir évaluation) |
| Context Recall | n/a | Incompatibilité RAGAS/Groq (voir évaluation) |

→ [Détails de l'évaluation](evaluation/results/ragas_scores.json)

---

## 🔄 Améliorations identifiées

- [ ] BGE-M3 pour hybrid retrieval dense+sparse (nécessite 16 Go+ RAM)
- [ ] Ground_truth vérifiées sur Légifrance
- [ ] OpenAI GPT-4o-mini comme LLM évaluateur RAGAS
- [ ] Mise à jour automatique du corpus (dump DILA mensuel)
- [ ] Extension à d'autres codes (Code civil, Code pénal)

---

## 📚 Notebooks

| Notebook | Description |
|---|---|
| [01_ingestion](notebooks/01_ingestion.ipynb) | Parsing XML LEGI, filtrage états juridiques |
| [02_chunking](notebooks/02_chunking.ipynb) | Stratégie adaptative, analyse distribution |
| [03_indexing_rag](notebooks/03_indexing_rag.ipynb) | Embeddings, t-SNE, test retrieval |
| [04_evaluation](notebooks/04_evaluation.ipynb) | Évaluation RAGAS, scores, analyse dataset |
| [05_monitoring](notebooks/05_monitoring.ipynb) | Monitoring Langfuse, traces, latence |
---

## 👤 Autrice

Projet réalisé afin d'explorer toutes les phases d'un projet de RAG

→ [Architecture complète](docs/ARCHITECTURE.md) | [Corpus](data/processed/CORPUS.md)


