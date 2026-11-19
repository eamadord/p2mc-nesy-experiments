"""
Mapeo de categorías de modelos KGC a la taxonomía de P2MC.

Bug corregido: antes la clave se construía con `lower().replace(" ", "-")`,
de modo que la etiqueta del clasificador "Semantic matching models" se
convertía en "semantic-matching-models" y no coincidía con la clave
"semantic-matching-model". Resultado: "Semantic", "Translation" y "Other"
caían siempre en el fallback "Other KGC technologies".

Ahora se canonicaliza cualquier texto (etiqueta del clasificador, ground
truth de PwC o salida libre de un LLM) a una de las cinco claves canónicas.
"""
import re

BASE_CATEGORY_URI = "https://w3id.org/p2mc/category"

# Las etiquetas "label" coinciden EXACTAMENTE con las clases del clasificador
# TF-IDF + SVC y con el ground truth de pwc_final.json.
CATEGORIES = {
    "semantic-matching": {
        "@id": f"{BASE_CATEGORY_URI}/semantic-matching",
        "name": "Semantic matching models",
    },
    "translation": {
        "@id": f"{BASE_CATEGORY_URI}/translation",
        "name": "Translation models",
    },
    "internal-side-information": {
        "@id": f"{BASE_CATEGORY_URI}/internal-side-information",
        "name": "Internal side information inside KGs",
    },
    "external-extra-information": {
        "@id": f"{BASE_CATEGORY_URI}/external-extra-information",
        "name": "External extra information outside KGs",
    },
    "other": {
        "@id": f"{BASE_CATEGORY_URI}/other",
        "name": "Other models",
    },
}

# Etiquetas que se le ofrecen al LLM como opciones cerradas.
CATEGORY_LABELS = [c["name"] for c in CATEGORIES.values()]


def canonical_category(text):
    """Devuelve la clave canónica ('translation', 'other', ...) o None si no se reconoce."""
    if not text:
        return None
    s = str(text).lower()
    s = re.sub(r"[^a-z]+", " ", s)
    if "semantic" in s or "matching" in s:
        return "semantic-matching"
    if "translation" in s or "translational" in s:
        return "translation"
    if "internal" in s:
        return "internal-side-information"
    if "external" in s or "extra information" in s:
        return "external-extra-information"
    if "other" in s:
        return "other"
    return None


class CategoryMapper:
    BASE_CATEGORY_URI = BASE_CATEGORY_URI
    CATEGORIES = CATEGORIES

    def get_category_object(self, extracted_category):
        key = canonical_category(extracted_category)
        if key is None:
            # El modelo devolvió algo fuera de la taxonomía: fallback explícito.
            key = "other"
        return dict(CATEGORIES[key])


# =============================================================================
# Conjunto de etiquetas genérico para el campo cerrado (categoría / área)
# =============================================================================
def _norm_label(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


class LabelSet:
    """
    Vocabulario cerrado para el campo "category" del pipeline.
      - perfil "kge":    las 5 categorías de modelos KGE (reglas de canonical_category)
      - perfil "hetero": las áreas de investigación de PwC (Computer Vision, NLP...)
    """

    def __init__(self, labels, base_uri, rules=None):
        self.labels = list(labels)
        self.base_uri = base_uri.rstrip("/")
        self._rules = rules
        self._by_norm = {_norm_label(l): l for l in self.labels}

    def canonical(self, text):
        """Devuelve la etiqueta oficial correspondiente a `text`, o None."""
        if not text:
            return None
        if self._rules is not None:
            key = self._rules(text)
            return CATEGORIES[key]["name"] if key else None
        n = _norm_label(text)
        if n in self._by_norm:
            return self._by_norm[n]
        from rapidfuzz import fuzz, process
        best = process.extractOne(n, list(self._by_norm), scorer=fuzz.ratio, score_cutoff=90)
        return self._by_norm[best[0]] if best else None

    def object(self, text, fallback=None):
        """Nodo JSON-LD {"@id", "name"} para la etiqueta, o None si no se reconoce."""
        if self._rules is not None:
            key = self._rules(text) or "other"
            return dict(CATEGORIES[key])
        label = self.canonical(text) or fallback
        if not label:
            return None
        return {"@id": f"{self.base_uri}/{_slug(label)}", "name": label}


KGE_LABELSET = LabelSet(CATEGORY_LABELS, BASE_CATEGORY_URI, rules=canonical_category)
