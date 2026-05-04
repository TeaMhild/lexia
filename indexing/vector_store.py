"""
indexing/vector_store.py

Module utilitaire pour charger et interroger l'index Chroma.

Distinction avec embedder.py :
- embedder.py  : script one-shot qui CONSTRUIT l'index
- vector_store.py : module qui CHARGE et INTERROGE l'index existant

Ce module est importé par rag/retriever.py, api/main.py
et les notebooks — jamais lancé directement.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_PATH  = "/workspaces/lexia/data/index/chroma"
MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"
COLLECTION  = "lexia_chunks"


# ── Singleton — charge le modèle une seule fois ───────────────────────────────
# NB : charger le modèle à chaque requête serait catastrophique
# pour les performances (~2s de chargement). Le singleton garantit que le
# modèle est chargé une seule fois au démarrage de l'application.

_model      = None
_client     = None
_collection = None


def get_model() -> SentenceTransformer:
    """Charge le modèle d'embedding une seule fois (singleton)."""
    global _model
    if _model is None:
        print(f"Chargement du modèle {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection() -> chromadb.Collection:
    """Charge la collection Chroma une seule fois (singleton)."""
    global _client, _collection
    if _collection is None:
        if not Path(INDEX_PATH).exists():
            raise FileNotFoundError(
                f"Index Chroma introuvable : {INDEX_PATH}\n"
                f"Lancez d'abord : python indexing/embedder.py"
            )
        _client = chromadb.PersistentClient(
            path=INDEX_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_collection(COLLECTION)
        print(f"Index chargé : {_collection.count()} chunks")
    return _collection


# ── Recherche sémantique ──────────────────────────────────────────────────────

def similarity_search(
    query: str,
    n_results: int = 5,
    filter_code: str = None,
) -> list[Document]:
    """
    Recherche sémantique dans l'index Chroma.

    Args:
        query       : question en langage naturel
        n_results   : nombre de chunks à retourner
        filter_code : filtrer par code juridique
                      ex: "Code du travail" ou "Code de la consommation"

    Returns:
        List[Document] triés par pertinence décroissante

    NB : le metadata filtering permet de restreindre
    la recherche à un code spécifique sans reconstruire l'index —
    c'est une des valeurs clés des métadonnées embarquées dès l'ingestion.
    """
    model      = get_model()
    collection = get_collection()

    # Encode la question
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    # Filtre optionnel par code juridique
    where = None
    if filter_code:
        where = {"code_name": filter_code}

    # Recherche dans Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Conversion en Documents LangChain
    documents = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Cosine distance → similarity score (0 à 1)
        score = round(1 - dist, 4)
        meta["relevance_score"] = score
        documents.append(Document(page_content=doc, metadata=meta))

    return documents


# ── Statistiques de l'index ───────────────────────────────────────────────────

def get_index_stats() -> dict:
    """
    Retourne les statistiques de l'index.
    Utilisé par le monitoring et l'endpoint GET /health de l'API.
    """
    collection = get_collection()

    # Échantillon pour les stats
    sample = collection.get(limit=1000, include=["metadatas"])
    metadatas = sample["metadatas"]

    from collections import Counter
    by_code = Counter(m.get("code_name", "unknown") for m in metadatas)
    by_type = Counter(m.get("chunk_type", "unknown") for m in metadatas)

    return {
        "total_chunks":  collection.count(),
        "index_path":    INDEX_PATH,
        "model":         MODEL_NAME,
        "by_code":       dict(by_code),
        "by_chunk_type": dict(by_type),
    }


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Test vector_store.py ─────────────────────────")

    # Stats
    stats = get_index_stats()
    print(f"\nIndex stats :")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Par code     : {stats['by_code']}")
    print(f"  Par type     : {stats['by_chunk_type']}")

    # Test similarity search
    queries = [
        ("Mon employeur peut-il me licencier pendant un arrêt maladie ?", None),
        ("Délai de rétractation achat en ligne", "Code de la consommation"),
        ("Rupture conventionnelle conditions", "Code du travail"),
    ]

    print("\n── Tests de recherche ───────────────────────────")
    for query, filter_code in queries:
        print(f"\nQ: {query}")
        if filter_code:
            print(f"   (filtré sur : {filter_code})")

        docs = similarity_search(query, n_results=3, filter_code=filter_code)
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] Art. {doc.metadata.get('article_num','?'):15}"
                  f" score={doc.metadata['relevance_score']:.3f}"
                  f" — {doc.page_content[:80]}...")