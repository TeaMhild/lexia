"""
indexing/embedder.py

Génère les embeddings avec paraphrase-multilingual-mpnet-base-v2
et les stocke dans Chroma.

Modèle choisi : paraphrase-multilingual-mpnet-base-v2
- 280Mo, supporte 50+ langues dont le français
- 768 dimensions, dense retrieval uniquement
- Contrainte : BGE-M3 (hybrid dense+sparse) nécessite 16Go+ RAM

Note architecture : en production avec 16Go+ RAM, migrer vers
BAAI/bge-m3 pour le hybrid retrieval dense+sparse.
"""

import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import psutil

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNKS_PATH = "/workspaces/lexia/data/processed/chunks.jsonl"
INDEX_PATH  = "/workspaces/lexia/data/index/chroma"
MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"

BATCH_SIZE  = 64    # sentence-transformers gère bien les batchs sur CPU, plus le BS est grand plus cela va vite
TEST_LIMIT  = None   # None = corpus complet / 500 pour un test


# ── Helpers ───────────────────────────────────────────────────────────────────

def ram_info() -> str:
    m = psutil.virtual_memory()
    return f"{m.available/1e9:.1f}Go dispo / {m.total/1e9:.1f}Go total"


# ── Chargement des chunks ─────────────────────────────────────────────────────

def load_chunks(path: str, limit: int = None) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks chargés")
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
) -> list[dict]:
    """
    Génère les embeddings pour chaque chunk.

    NB : sentence-transformers gère nativement
    le batching et la tokenisation — on lui passe juste les textes.
    normalize_embeddings=True est important pour la similarité cosine :
    ça permet d'utiliser le produit scalaire à la place de cosine
    (plus rapide dans Chroma).
    """
    texts = [c["page_content"] for c in chunks]

    print(f"\nEmbedding de {len(chunks)} chunks (batch_size={BATCH_SIZE})...")
    print(f"RAM avant : {ram_info()}")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # important pour cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start
    print(f"  Terminé en {elapsed:.1f}s ({len(chunks)/elapsed:.1f} chunks/s)")
    print(f"  RAM après : {ram_info()}")
    print(f"  Shape embeddings : {embeddings.shape}")

    return [
        {**chunk, "embedding": embeddings[i].tolist()}
        for i, chunk in enumerate(chunks)
    ]


# ── Stockage dans Chroma ──────────────────────────────────────────────────────

def store_in_chroma(
    embedded_chunks: list[dict],
    index_path: str,
) -> chromadb.Collection:
    """
    Stocke les embeddings dans Chroma (vector store local).

    NB : Chroma est idéal pour le développement —
    pas de serveur à lancer, persist sur disque, API simple.
    En production → Qdrant ou Weaviate pour la scalabilité.
    """
    Path(index_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=index_path,
        settings=Settings(anonymized_telemetry=False),
    )

    # Idempotent — supprime si existe déjà
    try:
        client.delete_collection("lexia_chunks")
        print("  Collection existante supprimée")
    except Exception:
        pass

    collection = client.create_collection(
        name="lexia_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nStockage dans Chroma ({len(embedded_chunks)} chunks)...")

    # Insertion par batch
    batch_size = 100
    for i in tqdm(range(0, len(embedded_chunks), batch_size), desc="Stockage"):
        batch = embedded_chunks[i:i + batch_size]

        ids        = [c["metadata"]["chunk_id"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        documents  = [c["page_content"] for c in batch]
        metadatas  = []

        for c in batch:
            # Chroma n'accepte que str/int/float/bool en metadata
            meta = {
                k: v for k, v in c["metadata"].items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"  {collection.count()} chunks indexés ✓")
    return collection


# ── Test retrieval ────────────────────────────────────────────────────────────

def test_retrieval(
    model: SentenceTransformer,
    collection: chromadb.Collection,
):
    """
    Test rapide pour vérifier la qualité du retrieval.
    Questions typiques d'un utilisateur LexIA.
    """
    test_queries = [
        "Mon employeur peut-il me licencier pendant un arrêt maladie ?",
        "Quel est le délai de rétractation pour un achat en ligne ?",
        "Quelles sont les conditions pour une rupture conventionnelle ?",
        "Quelles sont mes droits en cas de harcèlement au travail ?",
    ]

    print("\n── Test retrieval ───────────────────────────────────")
    for query in test_queries:
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\nQ: {query}")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - dist  # cosine distance → similarity
            print(f"  [{i+1}] Art. {meta.get('article_num','?'):15}"
                  f" ({meta.get('code_name','?'):25})"
                  f" score={score:.3f}")
            print(f"       {doc[:120]}...")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("LexIA — Phase 3 : Embedding & Indexing")
    print(f"Modèle : {MODEL_NAME}")
    print(f"Mode   : {'TEST (500 chunks)' if TEST_LIMIT else 'COMPLET'}")
    print(f"RAM    : {ram_info()}")
    print("=" * 55)

    # 1. Chargement chunks
    print("\n1. Chargement des chunks...")
    chunks = load_chunks(CHUNKS_PATH, limit=TEST_LIMIT)

    # 2. Modèle
    print("\n2. Chargement du modèle...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   RAM après modèle : {ram_info()}")

    # 3. Embedding
    print("\n3. Génération des embeddings...")
    embedded = embed_chunks(model, chunks)

    # 4. Stockage
    print("\n4. Stockage dans Chroma...")
    collection = store_in_chroma(embedded, INDEX_PATH)

    # 5. Test
    print("\n5. Test du retrieval...")
    test_retrieval(model, collection)

    print(f"\n✓ Phase 3 terminée !")
    print(f"  Index sauvegardé → {INDEX_PATH}")
    print(f"  RAM finale : {ram_info()}")


