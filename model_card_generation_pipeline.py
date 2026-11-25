"""
P2MC-NeSy: generación de model cards con ruta LLM o ruta CBR.

Estructura:
  * FieldPipeline: extrae los 6 campos evaluables de un paper por cuatro rutas
      - llm            : modelos grandes por campo (pipeline P2MC original)
      - cbr            : modelo pequeño + caso(s) recuperado(s)
      - small          : modelo pequeño SIN caso (ablación: ¿aporta algo el caso?)
      - retrieval_only : copia los campos del caso más similar, sin LLM (ablación)
    Todas las rutas usan las mismas entradas por campo y el mismo
    post-procesado simbólico (vocabulario cerrado del campo de área/categoría). Las tareas son abiertas.
  * ModelCardGenerator: pipeline de producción. Usa la ruta LLM hasta que la
    base de casos alcanza el umbral y la ruta CBR a partir de ahí; construye el
    JSON-LD y conserva (Retain) los casos generados.

Bugs corregidos respecto a la versión anterior:
  * La ruta LLM no extraía métricas; ahora ambas rutas extraen los mismos campos.
  * Los datasets de la ruta LLM solo recibían tablas (que además no se cargaban).
  * Las métricas se guardaban en "metrics" en la ruta CBR y en ningún sitio en la
    LLM; ahora van en hasEvaluation.evaluationMetrics en ambas.
  * referencePublication tenía un @id relativo ("1802.04394"); ahora es absoluto.
  * Se cargaban en GPU modelos que no se usaban (Qwen3-1.7B por transformers,
    varios GLiNER), lo que contaminaba la medición de recursos.
  * El mapeo de categorías casi siempre caía en "Other" (ver category_mapper.py).
"""
import copy
import json
import re
import time
from datetime import datetime

import joblib

import config
from extractors.field_extractor import FieldExtractor, Usage, make_field_specs
from utils.case_base import CaseBase, Embedder
from utils.evaluation_tools import normalize_metric
from utils.paper_loader import build_field_inputs
from utils.task_matcher import TaskMatcher
from utils.uri_builder import UriBuilder

EXTRACTED_FIELDS = ["name", "tasks", "category", "repo", "datasets", "metrics"]


def _empty_fields():
    return {"name": "", "category": "", "tasks": [], "tasks_raw": [], "repo": [], "datasets": [], "metrics": []}


class FieldPipeline:
    def __init__(self, generation="gen2026", client=None, profile=None, category_method=None,
                 classifier_path=config.TAXONOMY_CLASSIFIER):
        """
        profile: perfil de dataset (config.dataset_profile). Determina el vocabulario
                 del campo cerrado, la pregunta de ese campo y el vocabulario de tareas.
        """
        gen = config.GENERATIONS[generation]
        self.generation = generation
        self.profile = profile or config.dataset_profile()
        self.large_models = dict(gen["large"])
        self.small_model = gen["small"]
        self.category_method = category_method or self.profile["llm_category_method"]
        self.labelset = config.category_labelset(self.profile)
        specs = make_field_specs(self.labelset.labels, self.profile["category_question"])

        o = config.OLLAMA
        common = dict(client=client, host=o["host"], num_ctx=o["num_ctx"], num_predict=o["num_predict"],
                      keep_alive=o["keep_alive"], timeout=o["timeout"], specs=specs)
        self.large = FieldExtractor(self.large_models, **common)
        self.small = FieldExtractor({f: self.small_model for f in EXTRACTED_FIELDS}, **common)

        self._task_matcher = None      # solo para ENLAZAR tareas a URIs en el JSON-LD de producción
        self._classifier = None
        self._classifier_path = classifier_path

    # ------------------------------------------------------------------ utils
    def _classify_category(self, abstract):
        if self._classifier is None:
            self._classifier = joblib.load(self._classifier_path)
        return str(self._classifier.predict([abstract or ""])[0])

    @property
    def task_matcher(self):
        """Emparejador con el vocabulario de tareas: solo se usa para enlazar URIs en el JSON-LD, nunca en el experimento."""
        if self._task_matcher is None:
            self._task_matcher = TaskMatcher(self.profile["tasks_vocab"])
        return self._task_matcher

    @staticmethod
    def _clean_open_tasks(raw_tasks):
        """
        Las tareas son PREDICCIONES ABIERTAS: salen solo del texto del paper, sin cotejarlas con ningún
        vocabulario. Aquí solo se limpian espacios y se quitan duplicados exactos sin distinguir mayúsculas.
        """
        out, seen = [], set()
        for t in raw_tasks or []:
            t = " ".join(str(t).split())
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    def _canonical_category_label(self, label):
        return self.labelset.canonical(label) or ""

    # ----------------------------------------------------------------- rutas
    def run(self, route, paper, cases=None):
        """
        route: "llm" | "cbr" | "small" | "retrieval_only"
        paper: dict de utils.paper_loader.load_corpus()
        cases: lista de casos recuperados (solo cbr / retrieval_only)
        Devuelve (fields, usage_dict, wall_time_s)
        """
        t0 = time.perf_counter()
        usage = Usage()
        fields = _empty_fields()

        if route == "retrieval_only":
            if cases:
                src = cases[0]["fields"]
                for f in EXTRACTED_FIELDS:
                    fields[f] = copy.deepcopy(src.get(f, fields[f]))
        else:
            inputs = build_field_inputs(paper)
            extractor = self.large if route == "llm" else self.small
            use_cases = cases if route == "cbr" else None
            case_fields = [c["fields"] for c in (use_cases or [])]

            fields["name"] = extractor.extract("name", inputs["name"], case_fields, usage)
            fields["tasks_raw"] = extractor.extract("tasks", inputs["tasks"], case_fields, usage)
            fields["repo"] = extractor.extract("repo", inputs["repo"], case_fields, usage)
            fields["datasets"] = extractor.extract("datasets", inputs["datasets"], case_fields, usage)
            fields["metrics"] = extractor.extract("metrics", inputs["metrics"], case_fields, usage)

            if route == "llm" and self.category_method == "classifier":
                # Solo tiene sentido con la taxonomía KGE (es con la que se entrenó).
                fields["category"] = self._classify_category(paper.get("abstract"))
            else:
                fields["category"] = extractor.extract("category", inputs["category"], case_fields, usage)

            fields["tasks"] = self._clean_open_tasks(fields["tasks_raw"])

        fields["category"] = self._canonical_category_label(fields["category"])
        fields["repo"] = [u for u in fields["repo"] if re.match(r"^(https?://|www\.|github\.com|gitlab\.com)", u, re.I)]
        return fields, usage.as_dict(), time.perf_counter() - t0


# =============================================================================
# Pipeline de producción
# =============================================================================
class ModelCardGenerator:
    """
    Router dinámico: ruta LLM mientras la base de casos tenga menos de
    `cbr_threshold` casos; ruta CBR a partir de ahí. Los casos generados se
    conservan (Retain) en `case_dir`.
    """

    def __init__(self, cbr_threshold=500, generation="gen2026", case_dir="data/case_base",
                 k=1, retain_cbr_cases=False, enrich=True, resolve_uris=True, client=None,
                 embedder="minilm", dataset="kge"):
        self.cbr_threshold = cbr_threshold
        self.k = k
        self.case_dir = case_dir
        self.retain_cbr_cases = retain_cbr_cases
        self.enrich = enrich
        self.resolve_uris = resolve_uris

        self.profile = config.dataset_profile(dataset)
        self.fields = FieldPipeline(generation=generation, client=client, profile=self.profile)
        if embedder == "tfidf":
            raise ValueError("En producción usa 'minilm': el vectorizador TF-IDF no se persiste.")
        self.case_base = CaseBase(Embedder(embedder)).load(case_dir)
        print(f"[ModelCardGenerator] casos actuales: {len(self.case_base)}")

        with open(config.MODEL_TEMPLATE, encoding="utf-8") as f:
            self._template = json.load(f)
        self._uri_builder = UriBuilder()
        self._uri_fetcher = None
        self._summarizer = None

    # ---------------------------------------------------------------- helpers
    def _fetcher(self):
        if self._uri_fetcher is None:
            from utils.uri_fetcher import UriFetcher
            self._uri_fetcher = UriFetcher()
        return self._uri_fetcher

    def _metric_uri(self, metric):
        canon = normalize_metric(metric) or metric
        slug = re.sub(r"[^a-z0-9]+", "-", canon.lower()).strip("-") or "unknown-metric"
        return f"{UriBuilder.BASE_URI}/metric/{slug}"

    def _dataset_uri(self, dataset):
        uri = None
        if self.resolve_uris:
            try:
                uri = self._fetcher().guess_dataset_uri(dataset)
            except Exception:
                uri = None
        return uri or self._uri_builder.build_dataset_uri(dataset)

    def _author_uri(self, author, arxiv_id):
        uri = None
        if self.resolve_uris:
            try:
                uri = self._fetcher().extract_author_uri(author, arxiv_id)
            except Exception:
                uri = None
        return uri or self._uri_builder.build_author_uri(author)

    @staticmethod
    def _clean_empty(d):
        if isinstance(d, list):
            items = [ModelCardGenerator._clean_empty(v) for v in d]
            return [v for v in items if v not in (None, "", [], {})]
        if isinstance(d, dict):
            items = {k: ModelCardGenerator._clean_empty(v) for k, v in d.items()}
            return {k: v for k, v in items.items() if v not in (None, "", [], {})}
        return d

    # ---------------------------------------------------------------- JSON-LD
    def build_jsonld(self, paper, fields):
        arxiv_id = paper["arxiv_id"]
        jsonld = copy.deepcopy(self._template)
        jsonld["@id"] = self._uri_builder.build_modelcard_uri(arxiv_id)
        jsonld["dateCreated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        jsonld["name"] = fields.get("name") or "Unknown Model"

        # Las tareas se extraen abiertas. Aquí solo se intenta ENLAZAR cada una a una tarea conocida
        # (URI de Linked Papers With Code); si no encaja, se conserva con una URI propia en lugar de descartarla.
        tasks, seen_ids = [], set()
        for t in fields.get("tasks", []):
            m = self.fields.task_matcher.match_task(t)
            node = ({"@id": m["uri"], "name": m["name"]} if m else
                    {"@id": f"{UriBuilder.BASE_URI}/task/{re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')}", "name": t})
            if node["@id"] not in seen_ids:
                seen_ids.add(node["@id"])
                tasks.append(node)
        jsonld["mlTask"] = tasks
        jsonld.pop("modelCategory", None)
        cat = self.fields.labelset.object(fields.get("category"))
        if cat:
            jsonld[self.profile["jsonld_category_property"]] = cat
        jsonld["codeRepository"] = list(dict.fromkeys(fields.get("repo", [])))
        jsonld["evaluatedOn"] = [{"@id": self._dataset_uri(d), "name": d} for d in fields.get("datasets", [])]

        metrics = []
        for m in fields.get("metrics", []):
            uri = self._metric_uri(m)
            if uri not in {x["@id"] for x in metrics}:  # MR y Mean Rank -> una sola URI
                metrics.append({"@id": uri, "name": m})
        jsonld["hasEvaluation"] = {
            "@type": "fair4ml:MLModelEvaluation",
            "evaluationMetrics": metrics,
        }

        authors = paper.get("authors") or []
        jsonld["author"] = [{"@id": self._author_uri(a, arxiv_id), "name": a} for a in authors]

        if self.enrich:
            jsonld.update(self._enrichment(paper, jsonld["author"]))
        else:
            jsonld.pop("referencePublication", None)
            jsonld.pop("description", None)
            jsonld.pop("keywords", None)
        return self._clean_empty(jsonld)

    def _enrichment(self, paper, authors):
        """Descripción, keywords y publicación de referencia (no se evalúan ni se cronometran)."""
        out = {}
        if self._summarizer is None:
            from extractors.gemma_summarizer import GemmaSummarizer
            self._summarizer = GemmaSummarizer()
        abstract = paper.get("abstract", "")
        out["description"] = self._summarizer.summarize(abstract) or ""
        out["keywords"] = self._summarizer.get_keywords(abstract) or []

        arxiv_id = paper["arxiv_id"]
        ref = {
            "@type": "ScholarlyArticle",
            "@id": f"https://arxiv.org/abs/{arxiv_id}",   # URI absoluta (antes era relativa)
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "author": [{"@type": "Person", "@id": a["@id"]} for a in authors],
        }
        if self.resolve_uris:
            try:
                soa = self._fetcher().get_soa_data_from_arxiv_id(arxiv_id)
            except Exception:
                soa = None
            if soa:
                ref["schema:sameAs"] = soa.get("soa_uri")
                if soa.get("publicationDate"):
                    ref["datePublished"] = soa.get("publicationDate")
        out["referencePublication"] = ref
        return out

    # ------------------------------------------------------------------ router
    def generate_modelcard(self, paper):
        """
        paper: dict con arxiv_id, title, abstract, full_text, sections, tables_tsv, authors
        Devuelve (jsonld, info) donde info incluye la ruta usada, tiempos y uso de tokens.
        """
        if len(self.case_base) < self.cbr_threshold:
            route, cases = "llm", None
        else:
            route = "cbr"
            cases = [c for _, c in self.case_base.retrieve(paper.get("abstract"), k=self.k,
                                                           exclude_id=paper["arxiv_id"])]
        fields, usage, wall = self.fields.run(route, paper, cases)

        if route == "llm" or self.retain_cbr_cases:
            self.case_base.add({"arxiv_id": paper["arxiv_id"], "abstract": paper.get("abstract", ""),
                                "fields": fields, "source": route})
            self.case_base.save(self.case_dir)

        jsonld = self.build_jsonld(paper, fields)
        info = {"route": route, "extraction_wall_s": wall, **usage,
                "retrieved": [c["arxiv_id"] for c in (cases or [])]}
        return jsonld, info
