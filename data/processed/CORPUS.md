# Corpus LexIA — Documentation

## Source
- **Fournisseur** : DILA (Direction de l'Information Légale et Administrative)
- **Dataset** : Freemium LEGI — Codes, lois et règlements consolidés
- **Accès** : [data.gouv.fr](https://www.data.gouv.fr/datasets/legi-codes-lois-et-reglements-consolides)
- **Licence** : Licence Ouverte / Open Licence v2.0 (Etalab)
- **Date du dump** : 13 juillet 2025

## Téléchargement du dump

### 1. Trouver le dump le plus récent

Les dumps sont disponibles sur le serveur FTP de la DILA : https://echanges.dila.gouv.fr/OPENDATA/LEGI/
Les fichiers sont nommés `Freemium_legi_global_YYYYMMDD-HHMMSS.tar.gz`.
Prenez toujours le plus récent.

### 2. Télécharger le dump

```bash
# Remplacez la date par celle du dump le plus récent
curl -O "https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20250713-140000.tar.gz"
```

Le fichier fait ~1.1 Go — comptez 5 à 10 minutes selon votre connexion.

### 3. Extraire uniquement les deux codes

```bash
mkdir -p data/raw/legi_xml

tar -xzf Freemium_legi_global_20250713-140000.tar.gz \
  --wildcards \
  "legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/*LEGITEXT000006072050*" \
  "legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/*LEGITEXT000006069565*" \
  -C data/raw/legi/
```

> **Note** : l'extraction parcourt tout le tar.gz (1.1 Go) même si on ne garde
> que deux codes — comptez 2 à 3 minutes.

### 4. Vérifier l'extraction

```bash
# Doit retourner ~47 946 fichiers
find data/raw/legi -name "LEGIARTI*.xml" | wc -l

# Dont ~11 433 VIGUEUR pour le Code du travail
grep -rl "<ETAT>VIGUEUR</ETAT>" \
  data/raw/legi/global/code_et_TNC_en_vigueur/code_en_vigueur/LEGI/TEXT/00/00/06/07/20/LEGITEXT000006072050/article/ | wc -l
```

### 5. Lancer le pipeline d'ingestion

```bash
python ingestion/save_corpus.py
```

## Contenu

| Code | LEGITEXT | Articles retenus | États |
|---|---|---|---|
| Code du travail | LEGITEXT000006072050 | ~11 496 | VIGUEUR + VIGUEUR_DIFF |
| Code de la consommation | LEGITEXT000006069565 | ~2 148 | VIGUEUR + VIGUEUR_DIFF |
| **Total** | | **~13 644** | |

## Règles de filtrage

Seuls les articles avec les états suivants sont conservés :
- `VIGUEUR` : article actuellement en vigueur
- `VIGUEUR_DIFF` : article qui entrera en vigueur prochainement

Les états suivants sont exclus :
- `MODIFIE` : ancienne version remplacée par une plus récente
- `ABROGE` / `ABROGE_DIFF` : abrogé explicitement
- `PERIME` : périmé
- `ANNULE` : annulé par le Conseil d'État
- `TRANSFERE` : transféré dans un autre code
- `MODIFIE_MORT_NE` : modifié avant son entrée en vigueur

## Métadonnées par article

Chaque document embarque les métadonnées suivantes :

```json
{
  "source":       "legifrance_dump",
  "code_name":    "Code du travail",
  "article_id":   "LEGIARTI000020690617",
  "article_num":  "D6325-26",
  "section":      "Partie réglementaire > Sixième partie > Livre III > ...",
  "date_debut":   "2009-06-05",
  "date_fin":     "2999-01-01",
  "etat":         "VIGUEUR",
  "url":          "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000020690617"
}
```

## Limitations

- **Fraîcheur** : le corpus reflète l'état de la législation au **13 juillet 2025**. 
  Les modifications postérieures à cette date ne sont pas incluses.
- **Périmètre** : deux codes uniquement. Le pipeline est conçu pour être 
  étendu à n'importe quel code en ajoutant son LEGITEXT dans `loader.py`.
- **Exhaustivité** : 11 433 fichiers `VIGUEUR` identifiés par grep dans le dump,
  cohérent avec les 11 496 articles retenus par le parser (dont 72 `VIGUEUR_DIFF`).

## Mise à jour du corpus

Pour mettre à jour le corpus avec un dump plus récent :

```bash
# 1. Télécharger le nouveau dump sur echanges.dila.gouv.fr
# 2. Extraire les deux codes
# 3. Relancer le pipeline
python ingestion/save_corpus.py
```

## Structure des fichiers

```
data/
├── raw/
│   ├── legi/                          # Dump XML extrait (gitignore)
│   └── Freemium_legi_global_*.tar.gz  # Archive source (gitignore)
└── processed/
    ├── corpus.jsonl                   # 13 644 documents (gitignore)
    ├── corpus_metadata.json           # Statistiques et provenance
    └── CORPUS.md                      # Cette documentation
```