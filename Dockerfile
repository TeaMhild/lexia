# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.12 slim — image légère sans les outils de build inutiles
FROM python:3.12-slim

# ── Métadonnées ───────────────────────────────────────────────────────────────
LABEL maintainer="LexIA"
LABEL description="Assistant juridique RAG sur le droit français"
LABEL version="0.1.0"

# ── Variables d'environnement ─────────────────────────────────────────────────
# PYTHONUNBUFFERED : affiche les logs Python en temps réel (pas de buffering)
# PYTHONDONTWRITEBYTECODE : pas de fichiers .pyc inutiles dans le container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ── Répertoire de travail ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances système ───────────────────────────────────────────────────────
# curl : healthcheck
# build-essential : compilation de certains packages Python
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ────────────────────────────────────────────────────────
# On copie requirements.txt AVANT le code source
# Pourquoi : Docker cache les layers — si le code change mais pas les
# dépendances, Docker réutilise le layer pip install (build plus rapide)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY . .

# ── Installation du package ───────────────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker vérifie toutes les 30s que l'API répond bien
# Nb : le healthcheck permet à Docker/Kubernetes de
# redémarrer automatiquement le container s'il ne répond plus
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# ── Port exposé ───────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Commande de démarrage ─────────────────────────────────────────────────────
# --host 0.0.0.0 : écoute sur toutes les interfaces (obligatoire en container)
# --port $PORT   : port configurable via variable d'environnement
# workers 1      : un seul worker car le modèle d'embedding est en mémoire
#                  plusieurs workers = plusieurs copies du modèle = OOM
CMD uvicorn api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1