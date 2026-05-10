"""
evaluation/ragas_eval.py

Évaluation du système RAG LexIA avec le framework RAGAS.

Métriques utilisées :
- Faithfulness      : la réponse est-elle fidèle aux chunks retrieved ?
- Answer Relevancy  : la réponse répond-elle à la question ?
- Context Recall    : les chunks retrieved contiennent-ils l'info nécessaire ?
- Context Precision : les chunks retrieved sont-ils tous pertinents ?

Nb : RAGAS utilise un LLM pour évaluer les métriques
(LLM-as-judge) — on utilise le même Groq/Llama pour garder un coût nul.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import ask
from rag.retriever import retrieve_with_parent

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# On évalue sur un sous-ensemble pour limiter le coût
# Passer à None pour tout évaluer
MAX_SAMPLES = 10


# ── Chargement du dataset ─────────────────────────────────────────────────────

def load_eval_dataset(path: str, max_samples: int = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)
    if max_samples:
        dataset = dataset[:max_samples]
    print(f"Dataset chargé : {len(dataset)} questions")
    return dataset


# ── Génération des réponses ───────────────────────────────────────────────────

def generate_responses(dataset: list[dict]) -> list[dict]:
    """
    Pour chaque question, génère :
    - answer   : réponse de LexIA
    - contexts : chunks retrieved (liste de strings)

    Nb : on sépare la génération des réponses
    de l'évaluation RAGAS — on peut rejouer l'évaluation
    sans régénérer toutes les réponses (coûteux en temps).
    """
    results = []

    for i, item in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] {item['question'][:60]}...")

        # Retrieval
        docs = retrieve_with_parent(
            question=item["question"],
            n_results=5,
        )
        contexts = [doc.page_content[:500] for doc in docs]  # tronque à 500 chars
        # Génération
        answer = ask(
            question=item["question"],
            stream=False,
        )

        results.append({
            "question":    item["question"],
            "answer":      answer,
            "contexts":    contexts,
            "ground_truth": item["ground_truth"],
            "style":       item.get("style", "unknown"),
            "genre":       item.get("genre", "unknown"),
        })

        print(f"  Contexts : {len(contexts)} chunks")
        print(f"  Answer   : {answer[:100]}...")

    return results


# ── Évaluation RAGAS ──────────────────────────────────────────────────────────

def run_ragas_evaluation(results: list[dict]) -> dict:
    from ragas.run_config import RunConfig

    ragas_dataset = Dataset.from_dict({
        "question":     [r["question"] for r in results],
        "answer":       [r["answer"] for r in results],
        "contexts":     [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    ))

    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    ))

    # Désactive le parallélisme — Groq ne supporte pas n>1
    run_config = RunConfig(
        max_workers=1,      # séquentiel
        max_retries=3,
        timeout=60,
    )

    print("\nLancement évaluation RAGAS (séquentiel)...")
    scores = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    return scores



# ── Sauvegarde et analyse des résultats ──────────────────────────────────────

def save_and_analyze(results: list[dict], scores: dict):
    """
    Sauvegarde les résultats et analyse par style/genre.
    """
    # Sauvegarde des scores globaux
    scores_dict = {}
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        val = scores[metric]
        if isinstance(val, list):
            # Filtre les None et calcule la moyenne
            valid = [v for v in val if v is not None]
            scores_dict[metric] = sum(valid) / len(valid) if valid else None
        else:
            scores_dict[metric] = float(val) if val is not None else None

    output = {
        "scores_globaux": scores_dict,
        "nb_questions":   len(results),
        "details":        results,
    }

    output_path = RESULTS_PATH / "ragas_scores.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Affichage
    print("\n── Scores RAGAS ─────────────────────────────────────")
    print(f"Faithfulness      : {scores_dict['faithfulness']:.3f}")
    print(f"Answer Relevancy  : {scores_dict['answer_relevancy']:.3f}")
    print(f"Context Recall    : {scores_dict['context_recall']:.3f}")
    print(f"Context Precision : {scores_dict['context_precision']:.3f}")

    # Analyse par style
    print("\n── Analyse par style ────────────────────────────────")
    for style in ["formel", "informel"]:
        style_results = [r for r in results if r.get("style") == style]
        if style_results:
            print(f"  {style} : {len(style_results)} questions")

    # Analyse par genre
    print("\n── Analyse par genre ────────────────────────────────")
    for genre in ["masculin", "féminin", "neutre"]:
        genre_results = [r for r in results if r.get("genre") == genre]
        if genre_results:
            print(f"  {genre} : {len(genre_results)} questions")

    print(f"\nRésultats sauvegardés → {output_path}")
    return scores_dict


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("LexIA — Évaluation RAGAS")
    print("=" * 55)

    # 1. Chargement dataset
    dataset = load_eval_dataset(DATASET_PATH, max_samples=MAX_SAMPLES)

    # 2. Génération des réponses
    print("\n── Génération des réponses ──────────────────────────")
    results = generate_responses(dataset)

    # Sauvegarde intermédiaire — utile si RAGAS plante
    interim_path = RESULTS_PATH / "responses_interim.json"
    with open(interim_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRéponses sauvegardées → {interim_path}")

    # 3. Évaluation RAGAS
    scores = run_ragas_evaluation(results)

    # 4. Analyse et sauvegarde
    save_and_analyze(results, scores)

    print("\n✓ Évaluation terminée !")