---
title: LexIA API
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# LexIA — Assistant juridique RAG

Assistant juridique basé sur le droit français (Code du travail + Code de la consommation).

[![HuggingFace Space](https://img.shields.io/badge/🤗-Demo%20Live-blue)](https://huggingface.co/spaces/Majeylent/lexia-api)
[![API Docs](https://img.shields.io/badge/API-Swagger-green)](https://majeylent-lexia-api.hf.space/docs)


## API

- `POST /api/v1/query` — poser une question juridique
- `GET /api/v1/health` — état de l'API

## Stack

- **Embedding** : paraphrase-multilingual-mpnet-base-v2
- **Vector store** : Chroma
- **LLM** : Llama 3.3 70B via Groq API
- **Corpus** : 13 644 articles Légifrance (dump DILA, juillet 2025)

## Corpus

Le corpus est constitué de **13 644 articles juridiques** extraits du dump 
open data LEGI de la DILA (Légifrance), en vigueur au 13 juillet 2025.

→ [Documentation complète du corpus](data/processed/CORPUS.md)
