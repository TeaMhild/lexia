# Crée toute la structure d'un coup
mkdir -p ingestion indexing rag api/routes api/schemas \
         evaluation/results monitoring \
         data/raw data/processed data/index \
         notebooks tests .github/workflows

# Fichiers Python vides avec docstring
for f in ingestion/loader.py ingestion/cleaner.py ingestion/chunker.py \
          indexing/embedder.py indexing/vector_store.py \
          rag/retriever.py rag/chain.py rag/prompt.py \
          api/main.py monitoring/langfuse_client.py \
          evaluation/ragas_eval.py; do
  echo '"""'"${f}"'"""' > $f
done

# Fichiers data gitignorés mais dossiers trackés
touch data/raw/.gitkeep data/processed/.gitkeep data/index/.gitkeep

# .gitignore
cat > .gitignore << 'EOF'
.env
data/raw/*.jsonl
data/processed/*.jsonl
data/index/
__pycache__/
*.pyc
.DS_Store
EOF

# .env.example (jamais de vraies clés ici !)
cat > .env.example << 'EOF'
LEGIFRANCE_CLIENT_ID=your_client_id_here
LEGIFRANCE_CLIENT_SECRET=your_client_secret_here
OPENAI_API_KEY=your_openai_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_here
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
# Ingestion
pylegifrance
langchain
langchain-community

# Indexing & RAG
langchain-openai
chromadb
sentence-transformers

# API
fastapi
uvicorn[standard]

# Evaluation
ragas

# Monitoring
langfuse

# Utils
python-dotenv
tqdm
pytest
EOF

# devcontainer.json pour Codespaces
mkdir -p .devcontainer
cat > .devcontainer/devcontainer.json << 'EOF'
{
  "name": "LexIA",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt",
  "forwardPorts": [8000]
}
EOF

echo "Structure créée !"