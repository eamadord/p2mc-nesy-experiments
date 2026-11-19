"""
Extractor de campos unificado para las rutas LLM y CBR.

* Mismos prompts y mismas entradas para todas las rutas: lo único que cambia
  es el modelo y si se incluye el caso recuperado (CBR).
* Salida estructurada con esquema JSON (format=<schema>) en Ollama.
* Contabilidad de recursos por llamada a partir de la respuesta de Ollama:
  tokens de prompt y de salida, tiempo de cómputo y tiempo de carga del modelo
  (que se reporta aparte para no contaminar la comparación).
* MockOllamaClient permite probar el pipeline de punta a punta sin GPU.
"""
import json
import re
import time

from utils.category_mapper import CATEGORY_LABELS
from utils.evaluation_tools import clean_raw_llm_entity

# ---------------------------------------------------------------------------
# Prompts por campo. {context} = entrada del campo; {case} = bloque del caso.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise information extraction assistant for scientific papers. "
    "Use only the provided text. Never invent values. Answer only with JSON."
)

def make_field_specs(category_labels=None, category_question=None):
    """
    Especificación de cada campo. Los ejemplos son deliberadamente de varios dominios
    para no sesgar al modelo hacia KGE cuando el corpus es heterogéneo.
    category_labels: lista cerrada de etiquetas (categorías KGE o áreas de investigación).
    """
    labels = list(category_labels or CATEGORY_LABELS)
    question = category_question or "Which category best describes the proposed model?"
    return {
        "name": {
            "question": "What is the name of the main model or method PROPOSED in this paper "
                        "(not a baseline)? Return the short name used by the authors.",
            "key": "model_name",
            "type": "string",
        },
        "tasks": {
            "question": "Which machine learning tasks does this paper address "
                        "(e.g. Image Classification, Machine Translation, Link Prediction)?",
            "key": "tasks",
            "type": "list",
        },
        "category": {
            "question": f"{question} Choose exactly one of: " + "; ".join(labels) + ".",
            "key": "category",
            "type": "enum",
            "enum": labels,
        },
        "repo": {
            "question": "What is the URL of the official code repository of THIS paper "
                        "(e.g. GitHub)? Ignore links to datasets, other papers or third-party tools. "
                        "Return an empty list if there is none.",
            "key": "repository",
            "type": "list",
        },
        "datasets": {
            "question": "Which benchmark datasets are used to EVALUATE the proposed model "
                        "(e.g. ImageNet, SQuAD, FB15k-237)? Return dataset names only.",
            "key": "datasets",
            "type": "list",
        },
        "metrics": {
            "question": "Which evaluation metrics are reported (e.g. Accuracy, F1, BLEU, MRR)? "
                        "Return metric names only, without numeric values.",
            "key": "metrics",
            "type": "list",
        },
    }


FIELD_SPECS = make_field_specs()  # compatibilidad: taxonomía KGE por defecto

CASE_HEADER = (
    "Reference: values extracted from a SIMILAR paper. They may NOT apply to the target paper. "
    "Use them only as a guide for vocabulary and format; extract the values of the TARGET paper.\n"
)


def _schema(spec):
    if spec["type"] == "list":
        prop = {"type": "array", "items": {"type": "string"}}
    elif spec["type"] == "enum":
        prop = {"type": "string", "enum": spec["enum"]}
    else:
        prop = {"type": "string"}
    return {"type": "object", "properties": {spec["key"]: prop}, "required": [spec["key"]]}


def _case_block(field, cases):
    if not cases:
        return ""
    lines = []
    for i, case in enumerate(cases, 1):
        value = case.get(field)
        if value in (None, "", []):
            value = "(empty)"
        lines.append(f"Similar paper {i}: {json.dumps(value, ensure_ascii=False)}")
    return CASE_HEADER + "\n".join(lines) + "\n\n"


def build_messages(field, context, cases=None, specs=None):
    spec = (specs or FIELD_SPECS)[field]
    user = (
        f"{_case_block(field, cases)}"
        f"Target paper:\n<text>\n{context}\n</text>\n\n"
        f"Question: {spec['question']}\n"
        f"Answer with a JSON object with the single key \"{spec['key']}\"."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _get(obj, key, default=None):
    """Acceso compatible con dict y con los objetos de respuesta de ollama-python."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clean_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out, seen = [], set()
    for v in values:
        c = clean_raw_llm_entity(v)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


class Usage:
    """Acumulador de recursos de un paper."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.gen_tokens = 0
        self.compute_s = 0.0   # total_duration - load_duration
        self.load_s = 0.0      # carga del modelo en VRAM (se reporta aparte)
        self.failed_calls = 0

    def add(self, response, ok=True):
        self.calls += 1
        self.failed_calls += 0 if ok else 1
        if response is None:
            return
        total = (_get(response, "total_duration") or 0) / 1e9
        load = (_get(response, "load_duration") or 0) / 1e9
        self.prompt_tokens += _get(response, "prompt_eval_count") or 0
        self.gen_tokens += _get(response, "eval_count") or 0
        self.compute_s += max(0.0, total - load)
        self.load_s += load

    def as_dict(self):
        return {
            "llm_calls": self.calls,
            "llm_failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "llm_compute_s": round(self.compute_s, 4),
            "llm_load_s": round(self.load_s, 4),
        }


class FieldExtractor:
    """
    Extrae un campo con un modelo de Ollama.
    models: dict campo -> nombre del modelo de Ollama (p.ej. {"name": "llama3.1:8b", ...})
    """

    def __init__(self, models, client=None, host=None, num_ctx=16384, num_predict=256,
                 keep_alive="60m", timeout=600.0, max_retries=2, think=False, specs=None):
        self.models = models
        self.specs = specs or FIELD_SPECS
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self.think = think
        if client is None:
            from ollama import Client
            client = Client(host=host, timeout=timeout) if host else Client(timeout=timeout)
        self.client = client
        self._no_think_param = set()  # modelos que rechazan el parámetro `think`

    def _chat(self, model, messages, schema):
        kwargs = dict(
            model=model,
            messages=messages,
            format=schema,
            keep_alive=self.keep_alive,
            options={
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": 0.0,
                "seed": 0,
                "repeat_penalty": 1.0,
            },
        )
        if model not in self._no_think_param:
            try:
                return self.client.chat(think=self.think, **kwargs)
            except TypeError:
                self._no_think_param.add(model)   # cliente antiguo sin `think`
            except Exception as e:
                # Algunos modelos sin modo "thinking" devuelven error si se pasa think.
                if "think" in str(e).lower():
                    self._no_think_param.add(model)
                else:
                    raise
        return self.client.chat(**kwargs)

    def extract(self, field, context, cases=None, usage=None):
        spec = self.specs[field]
        model = self.models[field]
        messages = build_messages(field, context, cases, self.specs)
        schema = _schema(spec)

        response, parsed = None, None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._chat(model, messages, schema)
                content = _get(_get(response, "message"), "content", "") or ""
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                parsed = json.loads(content)
                break
            except json.JSONDecodeError:
                parsed = None
                break  # temperatura 0: reintentar daría la misma salida
            except Exception as e:
                msg = str(e).lower()
                if attempt < self.max_retries and any(k in msg for k in ("timeout", "timed out", "connection")):
                    time.sleep(2 * (attempt + 1))
                    continue
                print(f"[FieldExtractor] {field} / {model}: {e}")
                break

        if usage is not None:
            usage.add(response, ok=parsed is not None)

        value = parsed.get(spec["key"]) if isinstance(parsed, dict) else None
        if spec["type"] == "list":
            return _clean_list(value)
        if isinstance(value, list):
            value = value[0] if value else ""
        return clean_raw_llm_entity(value) if value else ""

    def loaded_models_vram(self):
        """VRAM que ocupa cada modelo cargado según `ollama ps` (bytes)."""
        try:
            resp = self.client.ps()
            models = _get(resp, "models") or []
            return {(_get(m, "name") or _get(m, "model")): _get(m, "size_vram") for m in models}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Backend simulado para pruebas sin GPU
# ---------------------------------------------------------------------------
class MockOllamaClient:
    """
    Imita client.chat() con heurísticas triviales y duraciones ficticias
    proporcionales a la longitud del prompt (modelos "pequeños" = más rápidos).
    SOLO para probar el pipeline; sus resultados no significan nada.
    """

    _KNOWN_METRICS = ["MRR", "MR", "Hits@1", "Hits@3", "Hits@10", "AUC", "AP", "Accuracy", "F1"]
    _KNOWN_DATASETS = ["FB15k-237", "FB15k", "WN18RR", "WN18", "YAGO3-10", "Cora", "Citeseer", "Pubmed", "UMLS"]

    def chat(self, model, messages, format=None, options=None, keep_alive=None, think=None):
        text = messages[-1]["content"]
        target = text.split("<text>", 1)[-1].split("</text>", 1)[0]
        key = list(format["properties"].keys())[0]
        if key == "model_name":
            m = re.search(r"\b([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9-]*)\b", target)
            val = m.group(1) if m else ""
        elif key == "category":
            enum = format["properties"]["category"]["enum"]
            val = enum[sum(map(ord, target[:200])) % len(enum)]
        elif key == "repository":
            val = re.findall(r"https?://github\.com/[\w.-]+/[\w.-]+", target)[:1]
        elif key == "datasets":
            val = [d for d in self._KNOWN_DATASETS if d in target]
        elif key == "metrics":
            val = [m for m in self._KNOWN_METRICS if re.search(rf"\b{re.escape(m)}\b", target)]
        else:  # tasks
            val = [t for t in ("Link Prediction", "Knowledge Graph Completion", "Node Classification")
                   if t.lower() in target.lower()]
        n_prompt = len(text) // 4
        speed = 4000 if any(s in model for s in ("0.8b", "1.5b", "2b")) else 1500  # tokens/s ficticios
        dur = int((n_prompt / speed + 0.05) * 1e9)
        return {
            "message": {"content": json.dumps({key: val})},
            "prompt_eval_count": n_prompt,
            "eval_count": 12,
            "total_duration": dur,
            "load_duration": 0,
        }

    def ps(self):
        return {"models": []}
