"""
ingestion/loader.py

Parse les fichiers XML du dump LEGI (DILA) et retourne
des Documents LangChain prêts pour le chunking.

Stratégie de filtrage des états :
- VIGUEUR : article en vigueur → ON GARDE
- VIGUEUR_DIFF : entrera en vigueur prochainement → ON GARDE
- MODIFIE : ancienne version remplacée par une plus récente → ON EXCLUT
- ABROGE / ABROGE_DIFF : abrogé explicitement → ON EXCLUT
- PERIME : périmé → ON EXCLUT
- ANNULE : annulé par le Conseil d'État → ON EXCLUT
- TRANSFERE : transféré dans un autre code → ON EXCLUT
- MODIFIE_MORT_NE : modifié avant entrée en vigueur → ON EXCLUT
"""

import re
from pathlib import Path
from xml.etree import ElementTree as ET
from langchain_core.documents import Document
from tqdm import tqdm
from cleaner import extract_contenu
# ── Configuration ─────────────────────────────────────────────────────────────

LEGI_BASE = Path("/workspaces/lexia/data/raw/legi")

CODES = {
    "LEGITEXT000006072050": "Code du travail",
    "LEGITEXT000006069565": "Code de la consommation",
}

# États à exclure — on garde tout le reste (VIGUEUR, VIGUEUR_DIFF)
ETATS_EXCLUS = {
    "MODIFIE",
    "ABROGE",
    "ABROGE_DIFF",
    "PERIME",
    "ANNULE",
    "TRANSFERE",
    "MODIFIE_MORT_NE",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_section_hierarchy(contexte: ET.Element) -> str:
    """
    Extrait la hiérarchie de sections depuis le CONTEXTE.
    Ex: "Partie législative > Titre II > Chapitre 1 > Section 3"
    
    Ces métadonnées permettent le metadata filtering
    au retrieval — ex: chercher uniquement dans la "Partie législative".
    """
    if contexte is None:
        return ""
    titles = []
    for tm in contexte.iter("TITRE_TM"):
        text = tm.text
        if text and text.strip():
            titles.append(text.strip())
    return " > ".join(titles)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_article(xml_path: Path, code_name: str) -> Document | None:
    """
    Parse un fichier XML LEGIARTI et retourne un Document LangChain.
    Retourne None si l'article doit être exclu.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ── Filtre état ───────────────────────────────────────────────────────
        etat = root.findtext("META/META_SPEC/META_ARTICLE/ETAT", "").strip()
        if etat in ETATS_EXCLUS:
            return None

        # ── Contenu textuel ───────────────────────────────────────────────────
        contenu = extract_contenu(root)
        if not contenu or len(contenu) < 20:
            return None

        # ── Métadonnées ───────────────────────────────────────────────────────
        article_id  = root.findtext("META/META_COMMUN/ID", "").strip()
        article_num = root.findtext("META/META_SPEC/META_ARTICLE/NUM", "").strip()
        date_debut  = root.findtext("META/META_SPEC/META_ARTICLE/DATE_DEBUT", "").strip()
        date_fin    = root.findtext("META/META_SPEC/META_ARTICLE/DATE_FIN", "").strip()

        contexte = root.find("CONTEXTE")
        section  = extract_section_hierarchy(contexte)

        return Document(
            page_content=contenu,
            metadata={
                "source":       "legifrance_dump",
                "code_name":    code_name,
                "article_id":   article_id,
                "article_num":  article_num,
                "section":      section,
                "date_debut":   date_debut,
                "date_fin":     date_fin,
                "etat":         etat,
                "url": f"https://www.legifrance.gouv.fr/codes/article_lc/{article_id}",
            },
        )

    except ET.ParseError as e:
        print(f"  Erreur parsing {xml_path.name}: {e}")
        return None


# ── Chargement par code ───────────────────────────────────────────────────────

def load_code(legitext_id: str, code_name: str) -> list[Document]:
    """
    Charge tous les articles d'un code depuis le dump XML.
    Cherche récursivement tous les fichiers LEGIARTI sous le dossier du code.
    """
    xml_files = [
        f for f in LEGI_BASE.rglob("LEGIARTI*.xml")
        if legitext_id in str(f)
    ]
    print(f"  {len(xml_files)} fichiers LEGIARTI trouvés")

    documents = []
    for xml_path in tqdm(xml_files, desc=code_name):
        doc = parse_article(xml_path, code_name)
        if doc:
            documents.append(doc)

    return documents


# ── Fonction principale ───────────────────────────────────────────────────────

def load_all_codes() -> list[Document]:
    """
    Point d'entrée principal : charge tous les codes définis dans CODES.
    Retourne la liste complète de Documents prêts pour le chunking.
    """
    all_documents = []

    for legitext_id, code_name in CODES.items():
        print(f"\nChargement : {code_name}")
        docs = load_code(legitext_id, code_name)
        print(f"  → {len(docs)} articles retenus")
        all_documents.extend(docs)

    print("\n── Statistiques corpus ──────────────────────────")
    print(f"Total documents  : {len(all_documents)}")
    if all_documents:
        avg_len = sum(len(d.page_content) for d in all_documents) // len(all_documents)
        print(f"Longueur moyenne : {avg_len} caractères")
        print(f"Codes chargés    : {set(d.metadata['code_name'] for d in all_documents)}")
        etats = {}
        for d in all_documents:
            e = d.metadata["etat"]
            etats[e] = etats.get(e, 0) + 1
        print(f"Répartition états: {etats}")

    return all_documents


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_all_codes()

    if docs:
        sample = docs[0]
        print("\n── Exemple document ─────────────────────────────")
        print(f"Contenu  : {sample.page_content[:300]}")
        print(f"Metadata : {sample.metadata}")