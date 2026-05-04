# Corpus LexIA — Documentation

LexIA est un assistant juridique RAG basé sur le droit français.
Ce document décrit la construction du corpus, les décisions techniques
et la procédure pour le reproduire de zéro.

## Table des matières

1. [Pipeline de construction](#pipeline-de-construction)
2. [Source des données](#source-des-données)
3. [Codes ingérés](#codes-ingérés)
4. [Filtrage des états juridiques](#filtrage-des-états-juridiques)
5. [Chunking](#chunking)
6. [Indexing](#indexing)
7. [Fichiers](#fichiers)
8. [Limitations](#limitations)
9. [Reproduire le corpus de zéro](#reproduire-le-corpus-de-zéro)

---

## Pipeline de construction

```
DILA (data.gouv.fr)          Source officielle open data
        │
        │ curl (1.1 Go tar.gz)
        ▼
Dump Freemium LEGI            ~47 946 fichiers XML LEGIARTI
        │
        │ tar --wildcards
        │ extraction 2 codes uniquement
        ▼
Fichiers XML LEGIARTI          Code du travail + Code de la consommation
        │
        │ ingestion/loader.py
        │ • filtrage ETAT (VIGUEUR / VIGUEUR_DIFF)
        │ • extraction contenu via itertext()
        │ • enrichissement métadonnées
        ▼
corpus.jsonl                   13 644 articles en vigueur
        │
        │ ingestion/chunker.py
        │ • stratégie adaptative par longueur
        │ • RecursiveCharacterTextSplitter
        │ • parent_content embarqué
        ▼
chunks.jsonl                   27 781 chunks
        │
        │ indexing/embedder.py
        │ • paraphrase-multilingual-mpnet-base-v2
        │ • normalize_embeddings=True
        │ • batch_size=64
        ▼
data/index/chroma/             Index vectoriel prêt pour le retrieval
```

---

## Source des données

| Champ | Valeur |
|---|---|
| Fournisseur | DILA (Direction de l'Information Légale et Administrative) |
| Dataset | Freemium LEGI — Codes, lois et règlements consolidés |
| Accès | [data.gouv.fr](https://www.data.gouv.fr/datasets/legi-codes-lois-et-reglements-consolides) |
| Licence | Licence Ouverte / Open Licence v2.0 (Etalab) |
| Date du dump | 13 juillet 2025 |

**Pourquoi le dump plutôt que l'API ?**
L'API Légifrance (portail PISTE) retourne **maximum 3 résultats par requête**
en accès gratuit — insuffisant pour un corpus exhaustif.
Le dump open data donne accès à l'intégralité des 73 codes en vigueur
sans limitation, est reproductible, et ne dépend pas d'une API tierce.

---

## Codes ingérés

| Code | LEGITEXT | Articles retenus |
|---|---|---|
| Code du travail | LEGITEXT000006072050 | 11 496 |
| Code de la consommation | LEGITEXT000006069565 | 2 148 |
| **Total** | | **13 644** |

---

## Filtrage des états juridiques

Le dump contient toutes les versions historiques de chaque article.
On conserve uniquement les articles en vigueur :

| État | Signification | Conservé |
|---|---|---|
| `VIGUEUR` | En vigueur | ✅ |
| `VIGUEUR_DIFF` | Entrera en vigueur prochainement | ✅ |
| `MODIFIE` | Ancienne version remplacée | ❌ |
| `ABROGE` | Abrogé explicitement | ❌ |
| `PERIME` | Périmé | ❌ |
| `ANNULE` | Annulé par le Conseil d'État | ❌ |
| `TRANSFERE` | Transféré dans un autre code | ❌ |
| `MODIFIE_MORT_NE` | Modifié avant son entrée en vigueur | ❌ |

### Exemple — article brut après ingestion

**Fichier source** : `LEGIARTI000020690617.xml`

**page_content** :
```
L'aide de l'Etat prévue aux articles D. 6325-23 et D. 6325-24 est attribuée
chaque année, en fonction du nombre d'accompagnements prévus par le groupement
d'employeurs. Elle est calculée sur une base forfaitaire par accompagnement
et par an, dont le montant est fixé par arrêté conjoint des ministres chargés
de l'emploi et du budget.
```

**metadata** :
```json
{
  "source":       "legifrance_dump",
  "code_name":    "Code du travail",
  "article_id":   "LEGIARTI000020690617",
  "article_num":  "D6325-26",
  "section":      "Partie réglementaire > Sixième partie > Livre III > Titre II > Chapitre V > Section 6",
  "date_debut":   "2009-06-05",
  "date_fin":     "2999-01-01",
  "etat":         "VIGUEUR",
  "url":          "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000020690617"
}
```

---

## Chunking

### Stratégie adaptative

| Catégorie | Seuil | Articles | chunk_size | overlap |
|---|---|---|---|---|
| Court | < 400 chars | ~8 270 (60%) | pas de découpage | — |
| Moyen | 400–10 000 chars | ~5 358 (39%) | 512 chars | 64 chars |
| Long (annexes) | > 10 000 chars | 16 (0.1%) | 256 chars | 32 chars |

**Splitter** : `RecursiveCharacterTextSplitter`
avec séparateurs `["\n\n", "\n", ".", ";", ",", " "]`

**Pourquoi Recursive** : respecte la ponctuation juridique — couper
au milieu d'un alinéa numéroté changerait le sens de l'article.

### Exemple — article découpé en chunks

Article `L1233-10` (892 chars) → découpé en 2 chunks :

**Chunk 0** (`chunk_type=standard`, 498 chars) :
```
L'employeur qui envisage de procéder à un licenciement collectif
pour motif économique d'au moins dix salariés dans une même période
de trente jours réunit et consulte le comité social et économique
dans les conditions prévues à la présente section...
```

**Chunk 1** (`chunk_type=standard`, 412 chars) :
```
...dans les conditions prévues à la présente section. Il indique :
1° La ou les raisons économiques, financières ou techniques du projet
de licenciement ; 2° Le nombre de licenciements envisagé ;
3° Les catégories professionnelles concernées...
```

**Overlap visible** : `dans les conditions prévues à la présente section`
apparaît à la fin du chunk 0 ET au début du chunk 1 — c'est l'overlap
de 64 chars qui évite de perdre la transition.

### Métadonnées ajoutées par chunk

```json
{
  "chunk_id":       "LEGIARTI000020690617_0",
  "chunk_index":    0,
  "chunk_total":    3,
  "chunk_type":     "standard",
  "parent_content": "texte complet de l'article parent..."
}
```

`parent_content` prépare le **parent-child retrieval** : on retrieve
sur les petits chunks (précision) mais on envoie l'article complet au LLM (contexte).

---

## Indexing

### Modèle d'embedding

| Paramètre | Valeur |
|---|---|
| Modèle | `paraphrase-multilingual-mpnet-base-v2` |
| Dimensions | 768 |
| Langues | 50+ dont français |
| Type | Dense uniquement |
| Taille | ~280 Mo |

**Pourquoi ce modèle et pas BGE-M3 ?**
BGE-M3 (hybrid dense+sparse, 4.56 Go) a été testé mais dépasse
la RAM disponible en environnement de développement (8.3 Go avec
4.7 Go déjà utilisés par le système). `paraphrase-multilingual-mpnet-base-v2`
offre un excellent compromis qualité/ressources pour le développement.
En production (16 Go+ RAM), BGE-M3 serait le choix recommandé pour
le hybrid retrieval dense+sparse.

### Vector Store

| Paramètre | Valeur |
|---|---|
| Moteur | Chroma (PersistentClient) |
| Distance | Cosine |
| Index | HNSW |
| Chunks indexés | 27 781 |

**Pourquoi Chroma ?**
Pas de serveur à lancer, persistence sur disque, API simple.
En production → Qdrant ou Weaviate pour la scalabilité et
le support natif du hybrid retrieval.

### Exemple — résultat d'une requête

**Question** : `"Mon employeur peut-il me licencier pendant un arrêt maladie ?"`

**Top 3 chunks retrouvés** :

| Rang | Article | Score | Extrait |
|---|---|---|---|
| 1 | L1226-9 | 0.821 | "Au cours des périodes de suspension du contrat de travail, l'employeur ne peut rompre ce dernier que s'il justifie..." |
| 2 | L1226-13 | 0.798 | "Toute rupture du contrat de travail prononcée en méconnaissance des dispositions des articles L. 1226-9..." |
| 3 | L1132-1 | 0.756 | "Aucune personne ne peut être écartée d'une procédure de recrutement ou de l'accès à un stage ou à une période de formation..." |

---

## Fichiers

```
data/processed/
├── CORPUS.md              # cette documentation ✅ versionné
├── corpus_metadata.json   # statistiques et provenance ✅ versionné
├── corpus.jsonl           # 13 644 articles bruts ❌ gitignored
└── chunks.jsonl           # chunks pour l'embedding ❌ gitignored

data/index/
└── chroma/                # index vectoriel Chroma ❌ gitignored
```
---

## Extension — Documents internes

L'architecture LexIA est conçue pour ingérer n'importe quelle source
documentaire sans modifier le chunker, l'embedder ou le vector store.
Seul le loader change selon le format source.

### Pourquoi c'est l'argument clé vs ChatGPT

ChatGPT ne connaît pas les documents internes d'une entreprise :
conventions collectives, accords d'entreprise, procédures RH, jurisprudences
internes. Un RAG peut les ingérer et les combiner avec le corpus Légifrance
pour des réponses contextualisées à l'entreprise.

### Loaders disponibles (LangChain)

| Format | Loader | Usage typique |
|---|---|---|
| PDF | `PyMuPDFLoader` | Conventions collectives, contrats |
| Word | `Docx2txtLoader` | Procédures RH, accords |
| Notion | `NotionDBLoader` | Base de connaissance interne |
| SharePoint | `OneDriveLoader` | Documents d'entreprise |
| CSV / Excel | `CSVLoader` | Grilles de salaires, tableaux |
| HTML / Web | `WebBaseLoader` | Intranet, FAQ en ligne |

### Intégration dans le pipeline

```python
# Exemple — ingestion d'une convention collective PDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

loader = PyMuPDFLoader("convention_collective.pdf")
docs = loader.load()

# Enrichissement des métadonnées — même structure que Légifrance
for doc in docs:
    doc.metadata.update({
        "source":      "interne",
        "doc_type":    "convention_collective",
        "entreprise":  "Acme Corp",
        "date":        "2024-01-15",
    })

# Même chunker, même embedder, même Chroma — rien ne change
```

### Metadata filtering au retrieval

La metadata `source` permet de cibler les recherches :

```python
# Cherche uniquement dans Légifrance
results = collection.query(
    query_embeddings=[query_embedding],
    where={"source": "legifrance_dump"},
    n_results=5,
)

# Cherche dans les deux corpus combinés
results = collection.query(
    query_embeddings=[query_embedding],
    where={"$or": [
        {"source": "legifrance_dump"},
        {"source": "interne"},
    ]},
    n_results=5,
)
```

### En production — pipeline incrémental

Pour un déploiement réel, on ajouterait :

1. **Surveillance des nouveaux documents** : watcher sur un dossier
   SharePoint ou une base Notion
2. **Indexing incrémental** : seuls les nouveaux documents sont indexés,
   pas tout le corpus
3. **Metadata `date_indexing`** : trace la fraîcheur de chaque document
4. **Déduplication** : évite d'indexer deux fois le même document

```python
# Squelette d'un pipeline incrémental
def index_if_new(doc: Document, collection) -> bool:
    existing = collection.get(ids=[doc.metadata["doc_id"]])
    if existing["ids"]:
        return False  # déjà indexé
    # sinon : chunk → embed → store
    return True
```

---

## Limitations

| Limitation | Impact | Solution envisagée |
|---|---|---|
| Fraîcheur figée au 13/07/2025 | Articles modifiés après cette date absents | Relancer le pipeline sur un dump plus récent |
| 2 codes sur 73 | Périmètre limité au droit du travail et de la consommation | Ajouter les LEGITEXT dans `loader.py` |
| Dense uniquement | Recherche par numéro d'article moins précise | Migrer vers BGE-M3 en production |
| Pas d'indexing incrémental | Réindexation complète si le corpus change | Implémenter un système de diff |
| ~60 min pour réindexer | Long en cas de changement | Indexing partiel par code |

---

## Reproduire le corpus de zéro

```bash
# 1. Télécharger le dump (~1.1 Go)
curl -O "https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20250713-140000.tar.gz"

# 2. Extraire les deux codes
mkdir -p data/raw/legi
tar -xzf Freemium_legi_global_20250713-140000.tar.gz \
  --wildcards \
  "legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/*LEGITEXT000006072050*" \
  "legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/*LEGITEXT000006069565*" \
  -C data/raw/legi/

# 3. Parser et sauvegarder le corpus
python ingestion/save_corpus.py
# → produit data/processed/corpus.jsonl (13 644 articles)

# 4. Chunker
python ingestion/chunker.py
# → produit data/processed/chunks.jsonl (27 781 chunks)

# 5. Indexer (~60 min sur 2 CPUs)
mkdir -p logs
nohup python indexing/embedder.py > logs/indexing.log 2>&1 &
tail -f logs/indexing.log
# → produit data/index/chroma/
```
