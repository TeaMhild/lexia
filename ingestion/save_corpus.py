"""
Sauvegarde le corpus en JSONL et ajoute les métadonnées globales.
"""

import json
from pathlib import Path
from langchain_core.documents import Document
from loader import load_all_codes

# ── Métadonnées globales du corpus ────────────────────────────────────────────
# Documenter la fraîcheur des données est critique est très importante dans ce use case
# pour un RAG juridique — les lois changent régulièrement.

CORPUS_METADATA = {
    "dump_date":   "2025-07-13",
    "source":      "DILA Légifrance Freemium LEGI",
    "disclaimer":  "Articles en vigueur au 13/07/2025 — peut ne pas refléter les modifications ultérieures",
    "codes":       ["Code du travail", "Code de la consommation"],
    "legitext_ids": ["LEGITEXT000006072050", "LEGITEXT000006069565"],
}

OUTPUT_PATH     = "/workspaces/lexia/data/processed/corpus.jsonl"
OUTPUT_META     = "/workspaces/lexia/data/processed/corpus_metadata.json"


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_documents(documents: list[Document], path: str) -> None:
    """
    Sauvegarde en JSONL — 1 document par ligne.
    Pourquoi JSONL : chargement ligne par ligne (faible RAM),
    facile à versionner, compatible avec tous les pipelines ML.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata":     doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Sauvegardé : {path} ({len(documents)} documents)")


def save_corpus_metadata(documents: list[Document], path: str) -> None:
    """
    Sauvegarde les métadonnées globales du corpus dans un fichier JSON séparé.
    Utile pour tracer la provenance et la fraîcheur des données.
    """
    from collections import Counter

    stats = {
        **CORPUS_METADATA,
        "total_documents": len(documents),
        "avg_length_chars": sum(len(d.page_content) for d in documents) // len(documents),
        "by_code": dict(Counter(d.metadata["code_name"] for d in documents)),
        "by_etat": dict(Counter(d.metadata["etat"] for d in documents)),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées : {path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# ── Chargement ────────────────────────────────────────────────────────────────

def load_documents(path: str) -> list[Document]:
    """Recharge les documents depuis le cache JSONL sans rappeler le parser."""
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(Document(
                page_content=record["page_content"],
                metadata=record["metadata"],
            ))
    return documents


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chargement du corpus depuis le dump XML...")
    docs = load_all_codes()

    print("\nSauvegarde...")
    save_documents(docs, OUTPUT_PATH)
    save_corpus_metadata(docs, OUTPUT_META)

    print("\nVérification du rechargement...")
    reloaded = load_documents(OUTPUT_PATH)
    print(f"Documents rechargés : {len(reloaded)} ✓")