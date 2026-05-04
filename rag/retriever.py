"""
rag/retriever.py

Récupère les chunks pertinents depuis le vector store
pour une question donnée.

Ce module fait le lien entre :
- vector_store.py  : accès à l'index Chroma
- chain.py         : génération de la réponse

NB : séparer le retriever de la chain permet
de tester et d'optimiser le retrieval indépendamment
de la génération — c'est une bonne pratique RAG.
"""

from langchain_core.documents import Document
from indexing.vector_store import similarity_search

# ── Configuration ─────────────────────────────────────────────────────────────

# Nombre de chunks retrieved — compromis entre contexte et coût LLM
# Trop peu : manque d'information
# Trop : contexte trop long, LLM se perd, coût plus élevé
DEFAULT_N_RESULTS = 5

# Seuil minimum de pertinence — on filtre les chunks trop éloignés
# En dessous de 0.5, le chunk est probablement hors sujet
MIN_RELEVANCE_SCORE = 0.58 
# Seuil 0.50 → trop permissif (météo passe)
# Seuil 0.65 → trop strict (rupture conventionnelle filtrée)
# Seuil 0.58 → meilleur compromis qu'on peut atteindre sans hybrid retrieval


# ── Retriever principal ───────────────────────────────────────────────────────

def retrieve(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_code: str = None,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[Document]:
    """
    Retrieve les chunks les plus pertinents pour une question.

    Args:
        question    : question en langage naturel
        n_results   : nombre de chunks à récupérer
        filter_code : restreindre à un code juridique spécifique
                      "Code du travail" ou "Code de la consommation"
        min_score   : score minimum de pertinence (0 à 1)

    Returns:
        List[Document] filtrés et triés par pertinence

    NB : le min_score est un levier important —
    trop bas on envoie des chunks hors sujet au LLM (hallucinations),
    trop haut on risque de ne rien retourner (réponse vide).
    0.5 est un bon point de départ à ajuster via l'évaluation RAGAS.
    """
    # Récupère les chunks depuis Chroma
    docs = similarity_search(
        query=question,
        n_results=n_results,
        filter_code=filter_code,
    )

    # Filtre par score minimum
    docs_filtered = [
        d for d in docs
        if d.metadata.get("relevance_score", 0) >= min_score
    ]

    # Log pour le debugging et le monitoring
    print(f"\n── Retrieval ────────────────────────────────────")
    print(f"Question : {question[:80]}")
    print(f"Chunks retrieved : {len(docs)} → après filtrage : {len(docs_filtered)}")
    for doc in docs_filtered:
        print(f"  Art. {doc.metadata.get('article_num','?'):15}"
              f" score={doc.metadata.get('relevance_score', 0):.3f}"
              f" — {doc.page_content[:60]}...")

    return docs_filtered


def retrieve_with_parent(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_code: str = None,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[Document]:
    """
    Variante parent-child : on retrieve sur les chunks (précision)
    mais on renvoie l'article parent complet au LLM (contexte).

    NB : c'est la stratégie parent-child retrieval —
    le petit chunk est bon pour trouver le bon article,
    mais le LLM a besoin du contexte complet pour répondre.
    On a anticipé ça en embarquant parent_content dès le chunking.
    """
    docs = retrieve(question, n_results, filter_code, min_score)

    # Remplace le contenu du chunk par l'article parent complet
    parent_docs = []
    seen_ids = set()

    for doc in docs:
        article_id = doc.metadata.get("article_id", "")

        # Déduplication — plusieurs chunks du même article
        # → on ne renvoie l'article qu'une seule fois
        if article_id in seen_ids:
            continue
        seen_ids.add(article_id)

        parent_content = doc.metadata.get("parent_content", doc.page_content)
        parent_doc = Document(
            page_content=parent_content,
            metadata=doc.metadata,
        )
        parent_docs.append(parent_doc)

    print(f"  Après déduplication parent : {len(parent_docs)} articles uniques")
    return parent_docs


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspaces/lexia")

    test_cases = [
        ("Mon employeur peut-il me licencier pendant un arrêt maladie ?", None),
        ("Quel est le délai de rétractation pour un achat en ligne ?", "Code de la consommation"),
        ("Quelles sont les conditions pour une rupture conventionnelle ?", "Code du travail"),
        ("Question hors sujet sur la météo en Bretagne", None),
    ]

    print("=" * 55)
    print("Test retrieve() — chunks")
    print("=" * 55)
    for question, filter_code in test_cases:
        docs = retrieve(question, filter_code=filter_code)
        print(f"  → {len(docs)} chunks retenus\n")

    print("=" * 55)
    print("Test retrieve_with_parent() — articles complets")
    print("=" * 55)
    question = "Mon employeur peut-il me licencier pendant un arrêt maladie ?"
    docs = retrieve_with_parent(question)
    print(f"\n  Article parent (premiers 300 chars) :")
    if docs:
        print(f"  {docs[0].page_content[:300]}...")