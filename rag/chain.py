"""
rag/chain.py

Connecte le retriever, le prompt et le LLM pour générer
une réponse juridique sourcée.

C'est le cœur du système RAG :
Question → Retriever → Prompt → LLM → Réponse

Modèle : llama-3.3-70b-versatile via Groq API
- Gratuit, très bon en français, contexte 128k tokens
- En production souveraine → remplacer par Mistral API
  ou Azure OpenAI (région EU) via LangChain en changeant
  uniquement l'initialisation du LLM
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
from monitoring.langfuse_client import trace_rag_query
from rag.prompt import get_prompt, format_context
from rag.retriever import retrieve_with_parent

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0.1   # faible pour des réponses factuelles et reproductibles
MAX_TOKENS   = 1024  # suffisant pour une réponse juridique complète


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """
    Initialise le LLM Groq.

    temperature=0.1 : on veut des réponses factuelles et reproductibles
    — pas de créativité dans un contexte juridique.

    NB : swapper le LLM se fait en changeant uniquement
    cette fonction — le reste de la chain ne change pas.
    C'est l'avantage de l'abstraction LangChain.

    Exemple migration vers Mistral :
    from langchain_mistralai import ChatMistralAI
    return ChatMistralAI(model="mistral-large-latest", temperature=0.1)
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Chain principale ──────────────────────────────────────────────────────────

def build_chain():
    """
    Construit la RAG chain avec LangChain Expression Language (LCEL).

    LCEL utilise le pipe operator | pour chaîner les composants :
    input → retriever → prompt → llm → output parser

    NB : LCEL est le standard LangChain depuis v0.2.
    Chaque composant est un Runnable — ils sont composables,
    streamables et observables (Langfuse en phase 6).
    """
    prompt = get_prompt()
    llm    = get_llm()

    chain = (
        {
            # Retrieval parallèle avec la question originale
            "context":  lambda x: format_context(
                            retrieve_with_parent(x["question"])
                        ),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Fonction principale ───────────────────────────────────────────────────────

def ask(
    question: str,
    filter_code: str = None,
    stream: bool = False,
) -> str:
    """
    Pose une question juridique à LexIA et retourne une réponse sourcée.

    Pipeline complet :
        1. Retrieve  — recherche sémantique dans Chroma (parent-child)
        2. Augment   — formate les articles retrieved en contexte LLM
        3. Generate  — appel Groq Llama 3.3 70B avec prompt strict
        4. Monitor   — trace automatique dans Langfuse (latence, chunks, réponse)

    Args:
        question    : question juridique en langage naturel
                      ex: "Mon employeur peut-il me licencier pendant un arrêt maladie ?"
        filter_code : restreint la recherche à un code spécifique
                      valeurs possibles : "Code du travail", "Code de la consommation"
                      None = recherche dans tout le corpus
        stream      : si True, affiche la réponse token par token (stdout)
                      utile pour l'API FastAPI avec SSE

    Returns:
        str : réponse juridique sourcée avec articles cités et URLs Légifrance.
              Contient toujours un disclaimer juridique en fin de réponse.
              Retourne un message d'indisponibilité si aucun article pertinent
              n'est trouvé (score < MIN_RELEVANCE_SCORE).

    Raises:
        Ne lève pas d'exception — les erreurs Langfuse sont non bloquantes.
        Les erreurs LLM ou Chroma se propagent normalement.

    Example:
        >>> response = ask(
        ...     question="Quel est le délai de rétractation pour un achat en ligne ?",
        ...     filter_code="Code de la consommation",
        ... )
        >>> print(response)
        Selon l'article L221-18 du Code de la consommation...

    NB :
        La séparation retrieve / augment / generate est le pattern RAG standard.
        Le monitoring Langfuse est non bloquant — une erreur de trace ne fait
        jamais planter la réponse utilisateur (graceful degradation).
    """
    llm    = get_llm()
    prompt = get_prompt()
    start  = time.time()

    # Retrieval
    docs    = retrieve_with_parent(question, filter_code=filter_code)
    context = format_context(docs)

    # Génération
    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    if stream:
        response = ""
        for chunk in llm.stream(messages):
            token = chunk.content
            print(token, end="", flush=True)
            response += token
        print()
    else:
        response = llm.invoke(messages).content

    latency_ms = (time.time() - start) * 1000

    # Trace dans Langfuse
    trace_rag_query(
        question=question,
        answer=response,
        contexts=[{
            "article_num":     d.metadata.get("article_num", ""),
            "code_name":       d.metadata.get("code_name", ""),
            "relevance_score": d.metadata.get("relevance_score", 0),
            "page_content":    d.page_content,
        } for d in docs],
        filter_code=filter_code,
        latency_ms=latency_ms,
    )

    return response

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("LexIA — Test RAG core")
    print("=" * 55)

    questions = [
        (
            "Mon employeur peut-il me licencier pendant un arrêt maladie ?",
            None,
        ),
        (
            "Quel est le délai de rétractation pour un achat en ligne ?",
            "Code de la consommation",
        ),
        (
            "Quelles sont les conditions pour une rupture conventionnelle ?",
            "Code du travail",
        ),
        (
            "Quel est le temps qu'il fait en Bretagne ?",
            None,
        ),
    ]

    for question, filter_code in questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        if filter_code:
            print(f"   (filtré sur : {filter_code})")
        print(f"{'='*55}")

        response = ask(question, filter_code=filter_code, stream=True)

        print(f"\n── Sources utilisées ────────────────────────────")
        docs = retrieve_with_parent(question, filter_code=filter_code)
        for doc in docs:
            print(f"  • Art. {doc.metadata.get('article_num','?'):15}"
                  f" {doc.metadata.get('url','')}")