"""
ingestion/cleaner.py

Fonctions de nettoyage du contenu textuel extrait du XML LEGI.

Séparation des responsabilités :
- loader.py   : charge et parse les fichiers XML
- cleaner.py  : nettoie le contenu textuel extrait
- chunker.py  : découpe les documents en chunks
"""

import re
from xml.etree import ElementTree as ET


def extract_contenu(root: ET.Element) -> str:
    """
    Extrait le texte du BLOC_TEXTUEL en récupérant tout le texte
    récursivement via itertext().

    Pourquoi itertext() et pas findtext() :
    Le contenu des articles est enveloppé dans des balises <p>, <br/>,
    <i> etc. findtext() retourne uniquement le texte direct de la balise
    (souvent '\n'). itertext() extrait récursivement tout le texte
    des balises enfants.
    """
    contenu_node = root.find("BLOC_TEXTUEL/CONTENU")
    if contenu_node is None:
        return ""
    contenu = " ".join(contenu_node.itertext())
    return clean_text(contenu)


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut extrait du XML LEGI.
    - Supprime les balises HTML résiduelles (<mark>, <b> etc.)
    - Normalise les espaces et sauts de ligne
    - Supprime les espaces en début/fin
    """
    text = re.sub(r"<[^>]+>", "", text)   # balises HTML résiduelles
    text = re.sub(r"\s+", " ", text)       # espaces multiples → un seul
    return text.strip()