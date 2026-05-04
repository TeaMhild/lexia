"""
rag/prompt.py

Templates de prompts pour LexIA.

Choix : prompt strict — le LLM répond uniquement depuis le contexte
fourni et indique explicitement quand l'information est insuffisante.

Pourquoi strict et pas augmenté :
- Contexte juridique = risque d'hallucination élevé
- Une hallucination sur un article de loi peut avoir des conséquences
  réelles pour l'utilisateur
- La transparence sur les limites est préférable à une réponse incorrecte

NB : c'est ce qu'on appelle le 'grounding' — ancrer
la réponse dans des sources vérifiables plutôt que dans la mémoire
du LLM.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Prompt système ────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """Tu es LexIA, un assistant juridique spécialisé
dans le droit français (Code du travail et Code de la consommation).

RÈGLES STRICTES :
1. Tu réponds UNIQUEMENT en te basant sur les articles juridiques
   fournis dans le contexte ci-dessous.
2. Pour chaque affirmation importante, cite l'article source
   avec son numéro et son URL Légifrance.
3. Si le contexte fourni ne contient pas d'information suffisante
   pour répondre à la question, réponds EXACTEMENT :
   "Je n'ai pas trouvé d'article suffisamment pertinent dans ma base
   pour répondre à cette question. Essayez de reformuler votre question
   avec des termes juridiques plus précis, ou consultez un professionnel."
4. Ne jamais inventer, extrapoler ou citer un article qui n'est pas
   dans le contexte fourni.
5. Réponds en français, de manière claire et accessible.
6. Termine toujours ta réponse par ce disclaimer :
   "⚠️ Cette réponse est fournie à titre informatif uniquement et ne
   constitue pas un conseil juridique. Pour votre situation personnelle,
   consultez un avocat ou un professionnel du droit."

CONTEXTE — Articles juridiques pertinents :
{context}
"""

HUMAN_TEMPLATE = "Question : {question}"

# ── Prompt template LangChain ─────────────────────────────────────────────────

def get_prompt() -> ChatPromptTemplate:
    """
    Retourne le prompt template LangChain.
    ChatPromptTemplate gère la séparation system/human
    attendue par les APIs de chat (OpenAI, Groq, Anthropic...).
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human",  HUMAN_TEMPLATE),
    ])


def format_context(documents) -> str:
    if not documents:
        return "Aucun article pertinent trouvé."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        # Tronque à 1500 chars pour éviter de dépasser les limites de tokens
        content = doc.page_content[:1500]
        if len(doc.page_content) > 1500:
            content += "... [tronqué]"

        part = (
            f"[Article {i}]\n"
            f"Code    : {meta.get('code_name', 'N/A')}\n"
            f"Numéro  : {meta.get('article_num', 'N/A')}\n"
            f"Section : {meta.get('section', 'N/A')[:100]}\n"
            f"URL     : {meta.get('url', 'N/A')}\n"
            f"Score   : {meta.get('relevance_score', 'N/A')}\n"
            f"Contenu :\n{content}\n"
        )
        context_parts.append(part)

    return "\n" + "─" * 50 + "\n".join(context_parts)

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_core.documents import Document

    # Simule des documents retrieved
    fake_docs = [
        Document(
            page_content="Au cours des périodes de suspension du contrat "
                         "de travail, l'employeur ne peut rompre ce dernier "
                         "que s'il justifie d'une faute grave ou d'une "
                         "impossibilité de maintenir le contrat.",
            metadata={
                "code_name":       "Code du travail",
                "article_num":     "L1226-9",
                "section":         "Titre II > Chapitre VI",
                "url":             "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006901177",
                "relevance_score": 0.821,
            }
        )
    ]

    prompt = get_prompt()
    context = format_context(fake_docs)

    print("── Prompt formaté ───────────────────────────────")
    print(prompt.format(
        context=context,
        question="Mon employeur peut-il me licencier pendant un arrêt maladie ?"
    ))


