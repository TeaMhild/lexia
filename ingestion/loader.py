"""
ingestion/loader.py

Récupère les articles juridiques depuis l'API Légifrance
via pylegifrance v1.6.3 et les retourne comme List[Document] LangChain.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from pylegifrance import LegifranceClient, ApiConfig
from pylegifrance.fonds.code import Code
from pylegifrance.models.code.enum import NomCode

load_dotenv()


# NomCode est un Enum — voici les deux codes qui nous intéressent
CODES = [
    (NomCode.CDT,  "Code du travail"),
    (NomCode.CDC,  "Code de la consommation"),
]
# ── Client Légifrance ────────────────────────────────────────────────────────

def get_client() -> LegifranceClient:
    """
    Initialise le client pylegifrance avec les clés OAuth du .env.
    ApiConfig lit automatiquement LEGIFRANCE_CLIENT_ID
    et LEGIFRANCE_CLIENT_SECRET depuis les variables d'environnement.
    """
    config = ApiConfig(
        client_id=os.getenv("LEGIFRANCE_CLIENT_ID"),
        client_secret=os.getenv("LEGIFRANCE_CLIENT_SECRET"),
    )
    return LegifranceClient(config)


# ── Transformation en Documents LangChain ────────────────────────────────────

def to_documents(articles: list, code_name: str) -> list[Document]:
    """
    Transforme les objets Article (Pydantic) en Documents LangChain.

    On embarque les métadonnées
    dès cette étape — elles permettront le metadata filtering
    au retrieval (ex: chercher uniquement dans le Code du travail).
    """
    documents = []

    for article in articles:
        texte = (article.texte or "").strip()

        # On filtre les articles vides ou trop courts (souvent abrogés)
        if not texte or len(texte) < 50:
            continue

        doc = Document(
            page_content=texte,
            metadata={
                "source":       "legifrance",
                "code_name":    code_name,
                "article_id":   article.id or "",
                "article_num":  article.num or "",
                "url": (
                    f"https://www.legifrance.gouv.fr/codes/article_lc/"
                    f"{article.id or ''}"
                ),
            },
        )
        documents.append(doc)

    return documents


# ── Fonction principale ───────────────────────────────────────────────────────

def load_all_codes() -> list[Document]:
    """
    Point d'entrée principal : charge tous les codes définis dans CODES.
    Retourne la liste complète de Documents prêts pour le chunking.
    """
    client = get_client()
    code_facade = Code(client)
    all_documents = []

    for nom_code, code_name in CODES:
        print(f"\nChargement : {code_name}")

        # Récupère tous les articles du code (état en vigueur)
        articles = (
            code_facade.search()
            .in_code(nom_code)
            .execute()
        )

        print(f"  → {len(articles)} articles bruts récupérés")

        docs = to_documents(articles, code_name)
        print(f"  → {len(docs)} documents valides après filtrage")

        all_documents.extend(docs)

    # Stats utiles à montrer 
    print("\n── Statistiques corpus ──────────────────────────")
    print(f"Total documents  : {len(all_documents)}")
    if all_documents:
        avg_len = sum(len(d.page_content) for d in all_documents) // len(all_documents)
        print(f"Longueur moyenne : {avg_len} caractères")
        codes = set(d.metadata["code_name"] for d in all_documents)
        print(f"Codes chargés    : {codes}")

    return all_documents


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_all_codes()

    if docs:
        sample = docs[0]
        print("\n── Exemple document ─────────────────────────────")
        print(f"Contenu  : {sample.page_content[:200]}...")
        print(f"Metadata : {sample.metadata}")