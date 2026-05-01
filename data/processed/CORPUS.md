# Corpus LexIA — Documentation

LexIA est un assistant juridique RAG basé sur le droit français.
Ce document décrit la construction du corpus, les décisions techniques
et la procédure pour le reproduire de zéro.

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
chunks.jsonl                   Prêts pour l'embedding (phase 3)
```

## Source des données

| Champ | Valeur |
|---|---|
| Fournisseur | DILA (Direction de l'Information Légale et Administrative) |
| Dataset | Freemium LEGI — Codes, lois et règlements consolidés |
| Accès | [data.gouv.fr](https://www.data.gouv.fr/datasets/legi-codes-lois-et-reglements-consolides) |
| Licence | Licence Ouverte / Open Licence v2.0 (Etalab) |
| Date du dump | 13 juillet 2025 |

### Pourquoi le dump plutôt que l'API ?

L'API Légifrance (portail PISTE) retourne **maximum 3 résultats par requête**
en accès gratuit — insuffisant pour un corpus exhaustif.
Le dump open data donne accès à l'intégralité des 73 codes en vigueur
sans limitation, est reproductible, et ne dépend pas d'une API tierce.

## Codes ingérés

| Code | LEGITEXT | Articles retenus |
|---|---|---|
| Code du travail | LEGITEXT000006072050 | 11 496 |
| Code de la consommation | LEGITEXT000006069565 | 2 148 |
| **Total** | | **13 644** |

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

## Métadonnées par article (corpus.jsonl)

```json
{
  "source":       "legifrance_dump",
  "code_name":    "Code du travail",
  "article_id":   "LEGIARTI000020690617",
  "article_num":  "D6325-26",
  "section":      "Partie réglementaire > Sixième partie > ...",
  "date_debut":   "2009-06-05",
  "date_fin":     "2999-01-01",
  "etat":         "VIGUEUR",
  "url":          "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000020690617"
}
```

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

### Métadonnées ajoutées par chunk (chunks.jsonl)

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

## Limitations

- **Fraîcheur** : corpus figé au **13 juillet 2025** — les modifications
  postérieures ne sont pas incluses
- **Périmètre** : 2 codes sur 73 disponibles dans le dump
- **Extensibilité** : ajouter un code = ajouter son LEGITEXT dans `loader.py`

## Fichiers

```
data/processed/
├── CORPUS.md              # cette documentation ✅ versionné
├── corpus_metadata.json   # statistiques et provenance ✅ versionné
├── corpus.jsonl           # 13 644 articles bruts ❌ gitignored
└── chunks.jsonl           # chunks pour l'embedding ❌ gitignored
```

## Reproduire le corpus de zéro

```bash
# 1. Télécharger le dump
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

# 4. Chunker
python ingestion/chunker.py
```