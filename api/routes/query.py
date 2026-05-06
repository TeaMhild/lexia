"""
api/routes/query.py

Endpoints de l'API LexIA :
- POST /query  : pose une question juridique
- GET  /health : vérifie l'état de l'API et de l'index
"""

from fastapi import APIRouter, HTTPException
from api.schemas.models import QueryRequest, QueryResponse, HealthResponse, ArticleSource
from rag.chain import ask
from rag.retriever import retrieve_with_parent
from indexing.vector_store import get_index_stats

router = APIRouter()


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Poser une question juridique",
    description="""
    Pose une question en langage naturel et reçoit une réponse
    sourcée depuis le Code du travail ou le Code de la consommation.

    La réponse cite les articles pertinents avec leurs URLs Légifrance.
    """,
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Nb : async est important ici — FastAPI est basé sur
    asyncio. Les endpoints async permettent de gérer plusieurs requêtes
    simultanées sans bloquer le serveur.

    Pour les opérations bloquantes (embedding, LLM), on utilise
    run_in_executor pour ne pas bloquer la boucle asyncio.
    """
    try:
        # Retrieval des articles pertinents
        docs = retrieve_with_parent(
            question=request.question,
            n_results=request.n_results,
            filter_code=request.filter_code,
        )

        # Génération de la réponse
        answer = ask(
            question=request.question,
            filter_code=request.filter_code,
            stream=False,
        )

        # Construction des sources citées
        sources = [
            ArticleSource(
                article_num=doc.metadata.get("article_num", ""),
                code_name=doc.metadata.get("code_name", ""),
                url=doc.metadata.get("url", ""),
                relevance_score=doc.metadata.get("relevance_score", 0.0),
                section=doc.metadata.get("section", ""),
            )
            for doc in docs
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            question=request.question,
            filter_code=request.filter_code,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement : {str(e)}",
        )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de l'API",
    description="Retourne le statut de l'API, le modèle utilisé et les stats de l'index.",
)
async def health() -> HealthResponse:
    """
    Endpoint de santé — appelé par les systèmes de monitoring
    et les load balancers pour vérifier que l'API est opérationnelle.

    Nb : un endpoint /health est indispensable en prod —
    il permet à Docker, Kubernetes ou Render de savoir si l'app
    est prête à recevoir des requêtes (readiness probe).
    """
    try:
        stats = get_index_stats()
        return HealthResponse(
            status="ok",
            model=stats["model"],
            total_chunks=stats["total_chunks"],
            by_code=stats["by_code"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service indisponible : {str(e)}",
        )


git add api/main.py api/routes/query.py api/schemas/models.py api/routes/__init__.py api/schemas/__init__.py
git commit -m "feat(phase4): FastAPI — endpoints POST /query + GET /health"
git push