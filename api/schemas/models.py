"""
api/schemas/models.py

Modèles Pydantic pour les requêtes et réponses de l'API LexIA.

NB : Pydantic garantit la validation automatique
des données entrantes — types, valeurs obligatoires, contraintes.
FastAPI s'appuie dessus pour générer la doc Swagger automatiquement.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Requête ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Question juridique en langage naturel",
        example="Mon employeur peut-il me licencier pendant un arrêt maladie ?",
    )
    filter_code: Optional[str] = Field(
        default=None,
        description="Restreindre la recherche à un code juridique spécifique",
        example="Code du travail",
    )
    n_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre de chunks à retriever",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Mon employeur peut-il me licencier pendant un arrêt maladie ?",
                "filter_code": "Code du travail",
                "n_results": 5,
            }
        }


# ── Source citée ──────────────────────────────────────────────────────────────

class ArticleSource(BaseModel):
    article_num:     str
    code_name:       str
    url:             str
    relevance_score: float
    section:         Optional[str] = None


# ── Réponse ───────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    answer:   str = Field(description="Réponse juridique générée par le LLM")
    sources:  list[ArticleSource] = Field(description="Articles juridiques utilisés")
    question: str = Field(description="Question posée")
    filter_code: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Selon l'article L1226-9...",
                "sources": [
                    {
                        "article_num":     "L1226-9",
                        "code_name":       "Code du travail",
                        "url":             "https://www.legifrance.gouv.fr/...",
                        "relevance_score": 0.821,
                        "section":         "Titre II > Chapitre VI",
                    }
                ],
                "question": "Mon employeur peut-il me licencier pendant un arrêt maladie ?",
                "filter_code": "Code du travail",
            }
        }


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model:        str
    total_chunks: int
    by_code:      dict


