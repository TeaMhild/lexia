# LexIA — Architecture du système

Ce document décrit l'architecture complète de LexIA, les décisions
techniques prises à chaque étape et les choix de déploiement.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Pipeline de données](#pipeline-de-données)
3. [RAG Core](#rag-core)
4. [API FastAPI](#api-fastapi)
5. [Déploiement](#déploiement)
6. [Décisions techniques clés](#décisions-techniques-clés)
7. [Limitations et évolutions](#limitations-et-évolutions)

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        LEXIA SYSTEM                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Corpus   │    │ Chunking │    │ Indexing │    │ RAG Core │  │
│  │ DILA     │───▶│ adaptatif│───▶│ Chroma   │───▶│ + LLM    │  │
│  │ 13 644   │    │ 27 781   │    │ 768 dims │    │ Groq     │  │
│  │ articles │    │ chunks   │    │ cosine   │    │ Llama 3  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                        │        │
│                                                        ▼        │
│                                               ┌──────────────┐  │
│                                               │  FastAPI     │  │
│                                               │  POST /query │  │
│                                               │  GET /health │  │
│                                               └──────────────┘  │
│                                                        │        │
│                                                        ▼        │
│                                               ┌──────────────┐  │
│                                               │  HuggingFace │  │
│                                               │  Spaces      │  │
│                                               │  Docker      │  │
│                                               └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline de données

### Phase 1 — Ingestion

**Source** : Dump Freemium LEGI — DILA (data.gouv.fr)
**Pourquoi** : L'API Légifrance (PISTE) retourne max 3 résultats/requête
en accès gratuit. Le dump open data donne accès aux 73 codes sans limite.

```
Dump XML LEGI (1.1 Go tar.gz)
        │
        │ tar --wildcards (extraction 2 codes)
        ▼
47 946 fichiers LEGIARTI*.xml
        │
        │ ingestion/loader.py
        │ • ET.parse() + itertext() — contenu dans balises <p>
        │ • Filtrage ETAT ∈ {VIGUEUR, VIGUEUR_DIFF}
        │ • Extraction hiérarchie sections (CONTEXTE/TM)
        │ • Enrichissement métadonnées
        ▼
13 644 Documents LangChain
        │
        │ ingestion/save_corpus.py
        ▼
corpus.jsonl
```

**Métadonnées par article** :
```json
{
  "source":       "legifrance_dump",
  "code_name":    "Code du travail",
  "article_id":   "LEGIARTI000020690617",
  "article_num":  "D6325-26",
  "section":      "Partie réglementaire > Sixième partie > ...",
  "date_debut":   "2009-06-05",
  "date_fin":     "2999-01-01",
  "etat":         "VIGUEUR",
  "url":          "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000020690617"
}
```

**Décision clé** : utilisation de `itertext()` plutôt que `findtext()` —
le contenu des articles est enveloppé dans des balises HTML `<p>`, `<br/>`.
`findtext()` retourne `'\n'`, `itertext()` extrait tout le texte récursivement.

---

### Phase 2 — Chunking

**Stratégie** : `RecursiveCharacterTextSplitter` adaptatif selon la longueur

```
corpus.jsonl (13 644 articles)
        │
        │ ingestion/chunker.py
        │
        ├── Court (< 400 chars, 60%) → 1 chunk = 1 article
        ├── Moyen (400-10k chars, 39%) → chunk_size=512, overlap=64
        └── Long (> 10k chars, 0.1%) → chunk_size=256, overlap=32
        ▼
chunks.jsonl (27 781 chunks)
```

**Pourquoi Recursive plutôt que Fixed-size** :
Les articles juridiques sont structurés en alinéas numérotés (1°, 2°, 3°...).
Le splitter Recursive coupe d'abord sur `\n\n`, puis `\n`, puis `.`, puis `;` —
il respecte la structure naturelle et évite de couper au milieu d'un alinéa.

**parent_content embarqué** : chaque chunk stocke le texte complet de l'article
parent pour le parent-child retrieval en phase 3.

**Distribution du corpus** :
- Min : 35 chars / Max : 112 726 chars (annexe technique)
- Médiane : 404 chars / P90 : 1 170 chars
- 16 articles outliers > 10 000 chars (annexes techniques)

---

### Phase 3 — Embedding & Indexing

**Modèle** : `paraphrase-multilingual-mpnet-base-v2`

**Pourquoi pas BGE-M3** : BGE-M3 (hybrid dense+sparse, 4.56 Go) testé
et confirmé OOM même en quantisation INT8 sur 8.3 Go RAM avec 4.7 Go
déjà utilisés par le système. Migration vers BGE-M3 documentée pour
la production (16 Go+ RAM).

```
chunks.jsonl (27 781 chunks)
        │
        │ indexing/embedder.py
        │ • SentenceTransformer (280 Mo, 768 dims)
        │ • normalize_embeddings=True (cosine similarity)
        │ • batch_size=64
        ▼
data/index/chroma/
        │
        │ indexing/vector_store.py (singleton)
        │ • get_model() — chargé une seule fois
        │ • get_collection() — chargé une seule fois
        │ • similarity_search(query, n_results, filter_code)
        ▼
Top-K chunks (filtrés par MIN_RELEVANCE_SCORE=0.58)
```

**Chroma** : vector store local, distance cosine, index HNSW.
En production → Qdrant on-premise pour la souveraineté des données.

---

## RAG Core

```
Question utilisateur
        │
        │ rag/retriever.py
        │ • encode la question (SentenceTransformer)
        │ • similarity_search(n_results=5)
        │ • filtre min_score=0.58
        │ • retrieve_with_parent() — article complet
        │ • déduplication par article_id
        ▼
Top-K articles pertinents
        │
        │ rag/prompt.py
        │ • format_context() — structure les articles
        │   (tronqué à 1500 chars/article — limite TPM Groq)
        │ • ChatPromptTemplate system + human
        │ • Prompt STRICT — grounding juridique
        ▼
Prompt enrichi
        │
        │ rag/chain.py
        │ • ChatGroq — llama-3.3-70b-versatile
        │ • temperature=0.1 (réponses factuelles)
        │ • streaming SSE disponible
        ▼
Réponse sourcée + articles cités + URLs Légifrance
+ disclaimer juridique systématique
```

### Prompt strict — grounding

Choix délibéré de ne pas laisser le LLM compléter avec ses connaissances
générales. Une hallucination sur un article de loi peut avoir des
conséquences réelles. Le système dit explicitement "je n'ai pas trouvé"
plutôt que de risquer une réponse incorrecte.

**Validé en test** : "Quel est le temps en Bretagne ?" → 0 chunks retenus
→ "Je n'ai pas trouvé d'article pertinent..."

### MIN_RELEVANCE_SCORE = 0.58

Ajusté empiriquement par itérations :
- 0.50 → trop permissif (articles sur intempéries passent pour "météo Bretagne")
- 0.65 → trop strict (rupture conventionnelle filtrée)
- 0.58 → compromis optimal

**À optimiser** via RAGAS en phase 5 sur un dataset de référence.

### LLM — Groq en développement, Mistral en production

| Environnement | LLM | Justification |
|---|---|---|
| Développement | Groq (Llama 3.3 70B) | Gratuit, rapide |
| Production souveraine | Mistral API | Entreprise française, données EU, DPA RGPD |
| Production Azure | Azure OpenAI | Région France Central, DPA RGPD |

Migration transparente via abstraction LangChain — une ligne dans `chain.py`.

---

## API FastAPI

```
POST /api/v1/query
├── Input  : QueryRequest (question, filter_code, n_results)
├── Process: retrieve_with_parent() → ask()
└── Output : QueryResponse (answer, sources, question)

GET /api/v1/health
├── Process: get_index_stats()
└── Output : HealthResponse (status, model, total_chunks, by_code)

GET /docs  → Swagger UI (auto-généré par FastAPI + Pydantic)
GET /      → métadonnées API
```

### Pattern Singleton + Lifespan

Le modèle d'embedding et l'index Chroma sont préchargés au démarrage
via le `lifespan` FastAPI. Sans ça, la première requête prendrait 3-4s.
Toutes les requêtes répondent en ~500ms dès le départ.

### CORS

`allow_origins=["*"]` en développement.
En production : restreindre aux domaines autorisés.

---

## Déploiement

### HuggingFace Spaces (actuel)

```
git push hf main
        │
        │ HuggingFace détecte Dockerfile
        │ Build Docker (~15 min)
        ▼
https://majeylent-lexia-api.hf.space/docs
```

**Specs** : CPU Basic — 2 vCPU, 16 Go RAM (gratuit)
**Limitation** : cold start après inactivité (~60s)

### Dockerfile

```dockerfile
FROM python:3.12-slim
# Dépendances → Code → pip install -e . → uvicorn
# Index Chroma inclus dans l'image (Option C)
# HEALTHCHECK toutes les 30s sur /api/v1/health
# 1 worker — modèle embedding en mémoire (pas de multi-process)
```

### Migration production (GCP Cloud Run)

```bash
gcloud run deploy lexia-api \
  --image gcr.io/mon-projet/lexia-api \
  --platform managed \
  --region europe-west1 \
  --memory 2Gi \
  --cpu 2 \
  --set-secrets GROQ_API_KEY=groq-api-key:latest
```

---

## Décisions techniques clés

| Décision | Choix | Alternative écartée | Raison |
|---|---|---|---|
| Source données | Dump DILA open data | API PISTE | Max 3 résultats/requête en gratuit |
| Parser XML | `itertext()` | `findtext()` | Contenu dans balises `<p>` |
| Chunking | RecursiveCharacterTextSplitter | Fixed-size | Respecte structure juridique |
| Embedding | paraphrase-multilingual | BGE-M3 | OOM sur 8.3 Go RAM |
| Vector store | Chroma | Qdrant | Dev local, pas de serveur |
| LLM | Groq Llama 3.3 70B | OpenAI GPT-4o | Gratuit |
| Prompt | Strict (grounding) | Augmenté | Risque hallucination juridique |
| Déploiement | HuggingFace Spaces | Render | 16 Go RAM gratuit vs 512 Mo |

---

## Limitations et évolutions

| Limitation | Impact | Solution production |
|---|---|---|
| Dense retrieval uniquement | Recherche par n° article moins précise | BGE-M3 hybrid sur 16 Go+ |
| Corpus figé 13/07/2025 | Articles modifiés absents | Pipeline mise à jour automatique |
| 2 codes sur 73 | Périmètre limité | Ajouter LEGITEXT dans loader.py |
| Troncature 1500 chars | Contexte parfois incomplet | LLM sans limite TPM stricte |
| MIN_SCORE empirique | Non optimal | Optimisation RAGAS phase 5 |
| Cold start HuggingFace | ~60s première requête | Instance dédiée ou keep-alive |
| Groq (données US) | Non souverain | Mistral API ou Azure OpenAI EU |

---

## Structure du projet

```
lexia/
├── ingestion/           # Phase 1 — Data ingestion
│   ├── loader.py        # Parse XML LEGI → Documents LangChain
│   ├── cleaner.py       # Nettoyage texte (itertext, re.sub)
│   ├── chunker.py       # Chunking adaptatif 3 stratégies
│   └── save_corpus.py   # Sauvegarde/rechargement JSONL
├── indexing/            # Phase 2-3 — Embedding & Indexing
│   ├── embedder.py      # Génère embeddings + stocke dans Chroma
│   └── vector_store.py  # Charge et interroge l'index (singleton)
├── rag/                 # Phase 3 — RAG Core
│   ├── retriever.py     # Retrieval + parent-child + min_score
│   ├── prompt.py        # Templates prompt juridique strict
│   └── chain.py         # Chain LangChain → Groq LLM
├── api/                 # Phase 4 — FastAPI
│   ├── main.py          # App FastAPI + lifespan + CORS
│   ├── routes/
│   │   └── query.py     # POST /query + GET /health
│   └── schemas/
│       └── models.py    # Pydantic models request/response
├── evaluation/          # Phase 5 — RAGAS (faithfulness=0.463)
├── monitoring/          # Phase 6 — Langfuse
├── notebooks/           # Exploration et démonstration
├── data/
│   ├── processed/
│   │   ├── CORPUS.md          # Documentation corpus
│   │   └── corpus_metadata.json
│   └── index/chroma/          # Index vectoriel
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

