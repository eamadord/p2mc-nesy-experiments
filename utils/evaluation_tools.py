"""
Herramientas de evaluación de P2MC-NeSy.

Cambios respecto a la versión anterior:
  * La evaluación principal es EXACT MATCH sobre entidades normalizadas
    (conjuntos -> P/R/F1). BERTScore sin reescalar devolvía ~0.8 incluso
    para cadenas sin relación y no discriminaba.
  * Normalizadores específicos por campo: métricas, datasets, URLs de
    repositorios, tareas, nombres de modelo y categorías.
  * BERTScore queda como métrica OPCIONAL solo para el nombre del modelo,
    con rescale_with_baseline=True y carga perezosa (antes se cargaba
    roberta-large en GPU solo con importar el módulo).
  * Se conservan clean_raw_llm_entity, normalize_entity,
    calculate_exact_match_f1 y aplanar por compatibilidad.
"""
import math
import re
import unicodedata

from utils.category_mapper import canonical_category

# ---------------------------------------------------------------------------
# Alias de métricas (claves ya normalizadas: minúsculas, sin espacios ni guiones)
# ---------------------------------------------------------------------------
ALIAS_MAP = {
    "mrr": "mrr",
    "meanreciprocalrank": "mrr",
    "meanreciprocalranking": "mrr",
    "mr": "mr",
    "meanrank": "mr",
    "f1": "f1",
    "f1score": "f1",
    "f1measure": "f1",
    "fscore": "f1",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "auc": "auroc",
    "auroc": "auroc",
    "rocauc": "auroc",
    "aucroc": "auroc",
    "areaundertheroccurve": "auroc",
    "auprc": "auprc",
    "aupr": "auprc",
    "prauc": "auprc",
    "aucpr": "auprc",
    "ap": "ap",
    "averageprecision": "ap",
    "map": "map",
    "meanap": "map",
    "meanaverageprecision": "map",
    "precision": "precision",
    "recall": "recall",
}

# Palabras de "configuración" que no cambian la métrica en sí.
_METRIC_NOISE_WORDS = r"\b(filtered|filter|filt|raw|test|valid|validation|ext)\b"


def aplanar(l):
    flat = []
    for i in l:
        if isinstance(i, (list, tuple)):
            flat.extend(aplanar(i))
        else:
            flat.append(str(i))
    return flat


def _strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# Limpieza bruta de salidas de LLM
# ---------------------------------------------------------------------------
def clean_raw_llm_entity(value):
    """Elimina citas, LaTeX y números sueltos de una entidad devuelta por un LLM."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"\[[^\]]{1,50}\]", "", s)
    s = re.sub(r"\((?:[^\)]*\d{4}[^\)]*|[^\)]*et\s*al\.?[^\)]*)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\$([^$]+?)\$", r"\1", s)
    s = re.sub(r"\\(?:textbf|textit|text|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\1", s)
    for _ in range(3):
        s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip(" .,:;-")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", s):
        return ""
    return s if len(s) >= 2 else ""


# ---------------------------------------------------------------------------
# Normalizadores por campo
# ---------------------------------------------------------------------------
def _compact(s):
    return re.sub(r"[-\s_/,.:;'\"]+", "", s)


def normalize_metric(entity):
    if not entity:
        return ""
    s = _strip_accents(str(entity).lower().strip())
    s = s.replace("%", "")
    s = re.sub(_METRIC_NOISE_WORDS, " ", s)

    # "Mean Reciprocal Rank (MRR)" -> probamos fuera y dentro del paréntesis
    inside = re.findall(r"\(([^)]*)\)", s)
    outside = re.sub(r"\([^)]*\)", " ", s)
    candidates = [outside] + inside + [s.replace("(", " ").replace(")", " ")]

    def _norm_one(x):
        x = _compact(x)
        x = x.replace("hitsat", "hits@").replace("hitat", "hits@")
        m = re.fullmatch(r"(?:hits?|h|hr|hitratio)@?(\d+)", x)
        if m:
            return f"hits@{m.group(1)}"
        m = re.fullmatch(r"(ndcg|map|mrr|p|precision|recall)@(\d+)", x)
        if m:
            return f"{m.group(1)}@{m.group(2)}"
        return ALIAS_MAP.get(x, x)

    normed = [_norm_one(c) for c in candidates if _compact(c)]
    if not normed:
        return ""
    known = set(ALIAS_MAP.values())
    for n in normed:
        if n in known or re.match(r"^(hits|ndcg|map|mrr|p|precision|recall)@\d+$", n):
            return n
    return normed[0]


def normalize_dataset(entity):
    if not entity:
        return ""
    s = _strip_accents(str(entity).lower().strip())
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # "wn18 (filtered)" -> "wn18"
    return _compact(s)


def normalize_entity(entity, is_metric=False):
    """Compatibilidad con la API anterior."""
    return normalize_metric(entity) if is_metric else normalize_dataset(entity)


def normalize_url(url):
    """
    https://github.com/Owner/Repo/tree/main -> github.com/owner/repo
    Para hosts de código conservamos host/owner/repo; para el resto,
    host + ruta sin barra final.
    """
    if not url:
        return ""
    s = str(url).strip().lower()
    s = s.strip("<>()[]{}\"' .,;")
    s = re.sub(r"^[a-z]+://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("#")[0].split("?")[0]
    s = re.sub(r"\.git$", "", s).rstrip("/")
    m = re.match(r"^(github\.com|gitlab\.com|bitbucket\.org)/([^/\s]+)/([^/\s]+)", s)
    if m:
        repo = re.sub(r"\.git$", "", m.group(3))
        return f"{m.group(1)}/{m.group(2)}/{repo}"
    return s


def normalize_task(task):
    if not task:
        return ""
    s = _strip_accents(str(task).lower().strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def normalize_name(name):
    if not name:
        return ""
    s = _strip_accents(str(name).lower().strip())
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # "TransC (bern)" -> "transc"
    s = re.sub(r"\b(model|framework|method|approach|the)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _name_tokens(name):
    s = _strip_accents(str(name).lower())
    s = re.sub(r"\(.*?\)", " ", s)
    stop = {"model", "the", "a", "an", "of", "for"}
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in stop]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def set_prf(predictions, ground_truth, normalizer):
    """
    P/R/F1 sobre conjuntos normalizados.
    Devuelve (nan, nan, nan) si el ground truth está vacío (campo no evaluable).
    """
    gt = {normalizer(g) for g in (ground_truth or []) if normalizer(g)}
    if not gt:
        return math.nan, math.nan, math.nan
    pred = {normalizer(p) for p in (predictions or []) if normalizer(p)}
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gt)
    p = tp / len(pred)
    r = tp / len(gt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def calculate_exact_match_f1(predictions, ground_truth, is_metric=False):
    """Compatibilidad con la API anterior (ambos vacíos -> 1.0)."""
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0
    p, r, f = set_prf(predictions, ground_truth, lambda x: normalize_entity(x, is_metric))
    if math.isnan(f):
        return 0.0, 0.0, 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def name_scores(prediction, ground_truth):
    """
    exact:    coincidencia exacta normalizada (0/1)
    relaxed:  una cadena normalizada contiene a la otra ('ConvE' vs 'ConvE model')
    token_f1: F1 a nivel de token (estilo SQuAD)
    """
    gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    gts = [g for g in gts if g and normalize_name(g)]
    if not gts:
        return {"exact": math.nan, "relaxed": math.nan, "token_f1": math.nan}
    pred = prediction[0] if isinstance(prediction, list) and prediction else prediction
    pn = normalize_name(pred) if pred else ""
    if not pn:
        return {"exact": 0.0, "relaxed": 0.0, "token_f1": 0.0}

    best = {"exact": 0.0, "relaxed": 0.0, "token_f1": 0.0}
    for g in gts:
        gn = normalize_name(g)
        exact = float(pn == gn)
        relaxed = float(bool(exact) or (len(pn) >= 3 and len(gn) >= 3 and (pn in gn or gn in pn)))
        pt, gt_ = _name_tokens(pred), _name_tokens(g)
        common = sum(min(pt.count(t), gt_.count(t)) for t in set(pt))
        if common == 0 or not pt or not gt_:
            tf1 = 0.0
        else:
            tp_, tr_ = common / len(pt), common / len(gt_)
            tf1 = 2 * tp_ * tr_ / (tp_ + tr_)
        for k, v in (("exact", exact), ("relaxed", relaxed), ("token_f1", tf1)):
            best[k] = max(best[k], v)
    return best


def category_accuracy(prediction, ground_truth, labelset=None):
    """Accuracy del campo cerrado. Sin labelset se usa la taxonomía KGE."""
    canon = labelset.canonical if labelset is not None else canonical_category
    gt = canon(ground_truth)
    if gt is None:
        return math.nan
    return float(canon(prediction) == gt)


# ---------------------------------------------------------------------------
# BERTScore opcional (solo nombre del modelo)
# ---------------------------------------------------------------------------
_bert_scorer = None


def bertscore_name(prediction, ground_truth):
    """BERTScore F1 reescalado con la línea base. Carga perezosa del modelo."""
    global _bert_scorer
    if not ground_truth:
        return math.nan
    if not prediction:
        return 0.0
    if _bert_scorer is None:
        import torch
        from bert_score import BERTScorer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
    pred = prediction[0] if isinstance(prediction, list) else prediction
    gt = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
    _, _, f1 = _bert_scorer.score([str(pred)], [str(gt)], verbose=False)
    return float(f1.item())


# ---------------------------------------------------------------------------
# Puntuación de un paper completo
# ---------------------------------------------------------------------------
FIELDS = ["name", "category", "tasks", "repo", "datasets", "metrics"]

# Métrica principal por campo (la que se usa en tablas y tests pareados).
PRIMARY_METRIC = {
    "name": "name_relaxed",
    "category": "category_acc",
    "tasks": "tasks_f1",
    "tasks_invocab": "tasks_invocab_f1",
    "repo": "repo_f1",
    "datasets": "datasets_f1",
    "metrics": "metrics_f1",
}


def score_paper(pred, gt, task_vocab=None, labelset=None):
    """
    pred: dict con name (str), category (str), tasks, repo, datasets, metrics (listas)
    gt:   dict con las mismas claves (ground truth)
    labelset:   vocabulario del campo cerrado (category_mapper.LabelSet); None = taxonomía KGE.
    task_vocab: nombres de tarea del vocabulario LPWC del TaskMatcher. Si se da, se
                añade 'tasks_invocab_*', que evalúa solo contra las tareas del GT
                que el sistema PUEDE llegar a predecir.
    Devuelve un dict plano {campo_metrica: valor}; NaN = campo no evaluable.
    """
    out = {}
    ns = name_scores(pred.get("name"), gt.get("name"))
    out["name_exact"], out["name_relaxed"], out["name_token_f1"] = ns["exact"], ns["relaxed"], ns["token_f1"]
    out["category_acc"] = category_accuracy(pred.get("category"), gt.get("category"), labelset)

    for field, normalizer in (
        ("tasks", normalize_task),
        ("repo", normalize_url),
        ("datasets", normalize_dataset),
        ("metrics", normalize_metric),
    ):
        p, r, f = set_prf(pred.get(field), gt.get(field), normalizer)
        out[f"{field}_p"], out[f"{field}_r"], out[f"{field}_f1"] = p, r, f

    if task_vocab is not None:
        vocab = {normalize_task(t) for t in task_vocab}
        gt_in = [t for t in (gt.get("tasks") or []) if normalize_task(t) in vocab]
        p, r, f = set_prf(pred.get("tasks"), gt_in, normalize_task)
        out["tasks_invocab_p"], out["tasks_invocab_r"], out["tasks_invocab_f1"] = p, r, f
    return out
