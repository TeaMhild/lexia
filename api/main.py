"""
api/main.py

Point d'entrée de l'API LexIA.

Démarre avec :
    uvicorn api.main:app --reload --port 8000

Nb : on sépare la création de l'app (main.py)
des routes (routes/) et des schemas (schemas/) — même logique
de séparation des responsabilités que pour le pipeline de données.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.query import router
from indexing.vector_store import get_model, get_collection


# ── Lifespan — chargement au démarrage ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager — exécuté au démarrage et à l'arrêt de l'app.

    Nb : on précharge le modèle d'embedding et l'index
    Chroma au démarrage plutôt qu'à la première requête.
    Sans ça, la première requête prendrait 3-4s (chargement modèle)
    — inacceptable en prod. Avec lifespan, toutes les requêtes
    répondent en ~500ms dès le départ.

    C'est le pattern "warm up" standard pour les APIs ML.
    """
    print("Démarrage LexIA API...")
    start = time.time()

    # Préchargement du modèle d'embedding (singleton)
    print("  Chargement du modèle d'embedding...")
    get_model()

    # Préchargement de l'index Chroma (singleton)
    print("  Chargement de l'index Chroma...")
    get_collection()

    elapsed = time.time() - start
    print(f"  API prête en {elapsed:.1f}s ✓")

    yield  # L'app tourne ici

    # Nettoyage à l'arrêt (optionnel)
    print("Arrêt LexIA API...")


# ── App FastAPI ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LexIA API",
    description="""
## Assistant juridique RAG sur le droit français

LexIA répond aux questions juridiques en s'appuyant sur :
- **Code du travail** — 11 496 articles en vigueur
- **Code de la consommation** — 2 148 articles en vigueur

### Comment ça marche ?
1. Votre question est encodée en vecteur (embedding)
2. Les articles les plus proches sémantiquement sont retrievés
3. Un LLM génère une réponse sourcée depuis ces articles

### Stack technique
- **Embedding** : paraphrase-multilingual-mpnet-base-v2
- **Vector store** : Chroma (HNSW, cosine similarity)
- **LLM** : Llama 3.3 70B via Groq API
- **Corpus** : dump open data DILA — 13 juillet 2025
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc (alternative à Swagger)
)


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # En prod : restreindre aux domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1", tags=["LexIA"])


# ── Endpoint racine ───────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée racine — redirige vers la doc."""
    return JSONResponse({
        "name":        "LexIA API",
        "version":     "0.1.0",
        "description": "Assistant juridique RAG sur le droit français",
        "docs":        "/docs",
        "health":      "/api/v1/health",
        "query":       "/api/v1/query",
    })



