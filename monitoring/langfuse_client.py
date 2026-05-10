"""
monitoring/langfuse_client.py
Langfuse v4 — API basée sur OpenTelemetry
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Initialisation ────────────────────────────────────────────────────────────

def init_langfuse():
    from langfuse import Langfuse
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("⚠️  Langfuse non configuré — monitoring désactivé")
        return None

    # Initialise avec les clés explicitement
    lf = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    return lf


langfuse = init_langfuse()


# ── Trace RAG ─────────────────────────────────────────────────────────────────

def trace_rag_query(
    question:    str,
    answer:      str,
    contexts:    list[dict],
    filter_code: str = None,
    latency_ms:  float = None,
    user_id:     str = None,
) -> str | None:
    if not langfuse:
        return None

    try:
        trace_id = langfuse.create_trace_id()

        # Span retrieval
        with langfuse.start_as_current_observation(
            name="retrieval",
            input={"question": question, "filter_code": filter_code},
        ):
            langfuse.update_current_span(
                output={
                    "n_chunks": len(contexts),
                    "chunks": [
                        {
                            "article_num":     c.get("article_num", ""),
                            "relevance_score": c.get("relevance_score", 0),
                        }
                        for c in contexts
                    ],
                    "avg_score": (
                        sum(c.get("relevance_score", 0) for c in contexts) / len(contexts)
                        if contexts else 0
                    ),
                },
            )

        # Span generation
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="generation",
            input=[{"role": "user", "content": question}],
            model="llama-3.3-70b-versatile",
            model_parameters={"temperature": 0.1, "max_tokens": 1024},
        ):
            langfuse.update_current_generation(
                output=answer,
                metadata={"latency_ms": latency_ms},
            )

        langfuse.flush()
        print(f"  Trace Langfuse : {trace_id}")
        return trace_id

    except Exception as e:
        print(f"  Erreur Langfuse (non bloquant) : {e}")
        return None

        
# ── Score feedback ────────────────────────────────────────────────────────────

def score_trace(trace_id: str, score: float, comment: str = None):
    """Ajoute un score de feedback sur une trace."""
    if not langfuse or not trace_id:
        return
    try:
        langfuse.create_score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment,
        )
        langfuse.flush()
        print(f"  Score ajouté sur trace {trace_id}")
    except Exception as e:
        print(f"  Erreur score Langfuse : {e}")


# ── Test connexion ────────────────────────────────────────────────────────────

def test_connection() -> bool:
    if not langfuse:
        return False
    try:
        langfuse.auth_check()
        print("✓ Connexion Langfuse OK")
        return True
    except Exception as e:
        print(f"✗ Erreur connexion Langfuse : {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test monitoring Langfuse...")

    if not test_connection():
        print("Vérifiez vos clés dans .env")
        exit(1)

    trace_id = trace_rag_query(
        question="Mon employeur peut-il me licencier pendant un arrêt maladie ?",
        answer="Selon l'article L1226-9, l'employeur ne peut pas...",
        contexts=[{
            "article_num":     "L1226-9",
            "code_name":       "Code du travail",
            "relevance_score": 0.821,
            "page_content":    "Au cours des périodes de suspension...",
        }],
        filter_code="Code du travail",
        latency_ms=1234,
    )

    if trace_id:
        print(f"✓ Trace créée : {trace_id}")
        print("  Vérifiez sur https://cloud.langfuse.com")
        score_trace(trace_id, score=1.0, comment="Test OK")