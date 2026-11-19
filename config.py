"""
Configuración central de modelos y experimentos de P2MC-NeSy.

"Generación" = conjunto de modelos de una época. Comparar el CBR con modelos
de 2024 y de 2026 permite mostrar que el resultado no depende de una
generación concreta de LLMs.

ANTES DE EJECUTAR: comprueba que los tags existen con `ollama list` /
`ollama pull <tag>` y anota en el paper la versión exacta (digest) de cada
modelo con `ollama show <tag> --modelfile`.
"""

GENERATIONS = {
    # Configuración original de los experimentos (modelos de 2024).
    "gen2024": {
        "large": {
            "name": "llama3.1:8b",
            "tasks": "llama3.1:8b",
            "repo": "llama3.1:8b",
            "datasets": "qwen2.5:7b",
            "metrics": "qwen2.5:7b",
            "category": "llama3.1:8b",   # solo si category_method == "llm"
        },
        "small": "qwen2.5:1.5b",
    },
    # Modelos actuales (2026). Qwen3.5 small: 0.8B/2B/4B/9B, en Ollama desde marzo de 2026.
    "gen2026": {
        "large": {
            "name": "qwen3.5:9b",
            "tasks": "qwen3.5:9b",
            "repo": "qwen3.5:9b",
            "datasets": "qwen3.5:9b",
            "metrics": "qwen3.5:9b",
            "category": "qwen3.5:9b",
        },
        "small": "qwen3.5:2b",
    },
}

# Campos que extrae el pipeline y que se evalúan. "category" es el campo de
# vocabulario cerrado: categoría de modelo KGE o área de investigación según el perfil.
FIELDS = ["name", "category", "tasks", "repo", "datasets", "metrics"]

# ---------------------------------------------------------------------------
# Perfiles de dataset
# ---------------------------------------------------------------------------
#   kge:    corpus original de Knowledge Graph Embedding (testing_data/)
#   hetero: corpus heterogéneo y completo construido con build_dataset.py
DATASETS = {
    "kge": {
        "data_dir": "testing_data",
        "category_field": "model_category",     # clave del GT en pwc_final.json
        "category_labels": "kge",               # taxonomía fija de category_mapper.py
        "category_question": "Which category best describes the proposed model?",
        "llm_category_method": "classifier",    # TF-IDF + SVC (P2MC original) o "llm"
        "jsonld_category_property": "modelCategory",
        "tasks_vocab": "testing_data/tasks.json",
        "n_test": 400,
        "thresholds": [100, 500, 1000, 1900],
    },
    "hetero": {
        "data_dir": "testing_data_hetero",
        "category_field": "area",
        "category_labels": "testing_data_hetero/areas.json",
        "category_question": "Which research area does this paper belong to?",
        "llm_category_method": "llm",           # el clasificador solo conoce la taxonomía KGE
        "jsonld_category_property": "about",    # un área no es un tipo de modelo: schema:about
        "tasks_vocab": "testing_data_hetero/tasks.json",
        "n_test": None,                         # la partición viene fijada por build_dataset.py
        "thresholds": [100, 500, 1000, 2000],
    },
}
DEFAULT_DATASET = "hetero"

# Parámetros de inferencia comunes (iguales para todas las rutas).
OLLAMA = {
    "host": None,          # None = OLLAMA_HOST o http://localhost:11434
    "num_ctx": 16384,
    "num_predict": 256,
    "keep_alive": "60m",   # evita recargas de modelo entre papers
    "timeout": 600.0,
}

# Diseño experimental.
# Los valores de DATASETS[<perfil>] (data_dir, n_test, thresholds...) tienen prioridad.
EXPERIMENT = {
    "out_dir": "results",      # se crea results/<perfil>/
    "split_seed": 13,
    "seeds": [0, 1, 2],        # ordenaciones aleatorias del pool
    "k": 1,                    # casos recuperados por consulta
    "embedder": "minilm",      # "minilm" (como ChromaDB) o "tfidf"
}

TAXONOMY_CLASSIFIER = "utils/taxonomy_classifier.joblib"
MODEL_TEMPLATE = "templates/template_1.jsonld"


def dataset_profile(name=None):
    """Perfil de dataset con la configuración del experimento ya combinada."""
    name = name or DEFAULT_DATASET
    prof = dict(EXPERIMENT)
    prof.update(DATASETS[name])
    prof["name"] = name
    return prof


def category_labelset(profile):
    """Conjunto de etiquetas del campo cerrado del perfil."""
    import json
    from utils.category_mapper import KGE_LABELSET, LabelSet
    spec = profile["category_labels"]
    if spec == "kge":
        return KGE_LABELSET
    with open(spec, encoding="utf-8") as f:
        labels = json.load(f)
    return LabelSet(labels, "https://linkedpaperswithcode.com/area")
