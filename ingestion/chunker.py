"""
ingestion/chunker.py

Stratégie de chunking adaptée au corpus juridique LEGI.

Trois cas selon la longueur de l'article :
- Court  (< 400 chars)    : 1 chunk = 1 article — pas de découpage
- Moyen  (400-10 000)     : RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
- Long   (> 10 000 chars) : même splitter mais chunk_size=256 pour les annexes techniques

On adapte la stratégie à la distribution réelle
des données plutôt que d'appliquer des paramètres fixes.
"""

import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ── Seuils ────────────────────────────────────────────────────────────────────

SHORT_THRESHOLD = 400    # chars — pas de découpage en dessous
LONG_THRESHOLD  = 10000  # chars — chunking agressif au dessus

# ── Splitters ─────────────────────────────────────────────────────────────────

# Splitter standard — articles moyens
# chunk_size=512 : bon compromis précision/contexte pour un modèle d'embedding
# chunk_overlap=64 : ~12% overlap pour ne pas perdre les transitions
splitter_standard = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", ";", ",", " "],
    length_function=len,
)

# Splitter agressif — annexes techniques longues
# chunk_size plus petit pour éviter des chunks trop hétérogènes
splitter_long = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=32,
    separators=["\n\n", "\n", ".", ";", ",", " "],
    length_function=len,
)


# ── Chunking d'un document ────────────────────────────────────────────────────

def chunk_document(doc: Document) -> list[Document]:
    """
    Découpe un Document en chunks selon sa longueur.
    Chaque chunk hérite des métadonnées du parent + chunk_id + chunk_index.

    On conserve l'article parent complet dans les
    métadonnées (parent_content) — utile pour le parent-child retrieval
    où on retrouve le petit chunk mais on envoie le grand au LLM.
    """
    content = doc.page_content
    length  = len(content)
    meta    = doc.metadata.copy()

    # ── Cas 1 : article court — pas de découpage ──────────────────────────────
    if length < SHORT_THRESHOLD:
        meta.update({
            "chunk_id":      f"{meta['article_id']}_0",
            "chunk_index":   0,
            "chunk_total":   1,
            "chunk_type":    "short",
            "parent_content": content,
        })
        return [Document(page_content=content, metadata=meta)]

    # ── Cas 2 : article long (annexe) — chunking agressif ────────────────────
    if length > LONG_THRESHOLD:
        splitter = splitter_long
        chunk_type = "annexe"
    # ── Cas 3 : article moyen — chunking standard ─────────────────────────────
    else:
        splitter = splitter_standard
        chunk_type = "standard"

    chunks = splitter.split_text(content)
    documents = []

    for i, chunk in enumerate(chunks):
        chunk_meta = meta.copy()
        chunk_meta.update({
            "chunk_id":       f"{meta['article_id']}_{i}",
            "chunk_index":    i,
            "chunk_total":    len(chunks),
            "chunk_type":     chunk_type,
            "parent_content": content,  # article complet pour parent-child retrieval
        })
        documents.append(Document(page_content=chunk, metadata=chunk_meta))

    return documents


# ── Fonction principale ───────────────────────────────────────────────────────

def chunk_corpus(
    input_path:  str = "/workspaces/lexia/data/processed/corpus.jsonl",
    output_path: str = "/workspaces/lexia/data/processed/chunks.jsonl",
) -> list[Document]:
    """
    Charge le corpus, chunk chaque document, sauvegarde en JSONL.
    """
    # Chargement
    print("Chargement du corpus...")
    raw_docs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            raw_docs.append(Document(
                page_content=record["page_content"],
                metadata=record["metadata"],
            ))
    print(f"  {len(raw_docs)} documents chargés")

    # Chunking
    print("\nChunking...")
    all_chunks = []
    stats = {"short": 0, "standard": 0, "annexe": 0}

    for doc in tqdm(raw_docs):
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        stats[chunks[0].metadata["chunk_type"]] += len(chunks)

    # Statistiques
    print(f"\n── Statistiques chunks ──────────────────────────")
    print(f"Total chunks     : {len(all_chunks)}")
    print(f"Ratio expansion  : {len(all_chunks)/len(raw_docs):.2f}x")
    print(f"Chunks courts    : {stats['short']}")
    print(f"Chunks standard  : {stats['standard']}")
    print(f"Chunks annexes   : {stats['annexe']}")

    chunk_lens = [len(c.page_content) for c in all_chunks]
    print(f"Longueur moyenne : {sum(chunk_lens)//len(chunk_lens)} chars")
    print(f"Longueur max     : {max(chunk_lens)} chars")

    # Sauvegarde
    print(f"\nSauvegarde → {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata":     chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sauvegardé : {len(all_chunks)} chunks")
    return all_chunks


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = chunk_corpus()

    # Exemple d'un chunk standard pour vérification
    sample = next(c for c in chunks if c.metadata["chunk_type"] == "standard")
    print(f"\n── Exemple chunk standard ───────────────────────")
    print(f"Contenu    : {sample.page_content[:200]}")
    print(f"Chunk ID   : {sample.metadata['chunk_id']}")
    print(f"Chunk type : {sample.metadata['chunk_type']}")
    print(f"Total dans cet article : {sample.metadata['chunk_total']}")
