"""
Carga del corpus de test y construcción de las entradas por campo.

Principio de diseño: cada campo recibe SIEMPRE la misma entrada, se use la
ruta LLM o la ruta CBR. Así la única diferencia entre rutas es el modelo y la
presencia (o no) del caso recuperado, y la comparación es justa.

Bug corregido: antes se buscaban las tablas en testing_data/lightocr_json
cuando la carpeta real es testing_data/lightocr_jsons, por lo que las tablas
nunca se cargaban.
"""
import json
import random
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from utils.XMLParser import XMLParser

# Secciones donde suelen aparecer datasets y métricas (coincidencia parcial,
# sin distinguir mayúsculas: "experiment" cubre "Experiments", "Experimental Setup"...)
EXPERIMENT_SECTIONS = ["experiment", "evaluation", "result", "setup", "dataset", "benchmark"]

# Presupuestos de caracteres por campo (~4 caracteres por token). Se trunca de
# forma explícita y documentada: Ollama trunca en silencio si el prompt supera num_ctx.
CHAR_BUDGET = {
    "name": 6000,
    "repo": 6000,
    "experiments": 16000,
    "tables": 12000,
}

_URL_HINT = re.compile(
    r"(https?://|www\.|github|gitlab|bitbucket|code (is|are) (publicly )?available|source code|implementation is)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tablas (LightOnOCR -> TSV compacto)
# ---------------------------------------------------------------------------
def _iter_ocr_tables(raw):
    """
    Tablas de un JSON de LightOnOCR, en cualquiera de sus tres formatos:
      A) {"file_name", "results": [{"page", "tables": [{"html": ...}]}]}            (JSON de testing_data/)
      B) {"documents": [{"paper_title", "tables": [{"page", "evaluation": {"columns"}, "rows"}]}]}
                                                                          (salida de LightOcrParser.process)
      C) {"file_name", "results": [{"page", "tables": [{"html", "columns", "rows"}]}]}  (fase ocr de build_dataset.py)
    Antes solo se leía A: el formato B daba 0 tablas sin avisar.
    """
    docs = raw.get("documents") if isinstance(raw, dict) and "documents" in raw else [raw]
    for doc in docs or []:
        if not isinstance(doc, dict):
            continue
        for page in doc.get("results") or []:
            for t in (page or {}).get("tables") or []:
                if isinstance(t, dict):
                    yield t
        for t in doc.get("tables") or []:
            if isinstance(t, dict):
                yield t


def html_tables_from_ocr(raw):
    """Fragmentos HTML de las tablas que los traen (compatibilidad con el formato A)."""
    return [t["html"] for t in _iter_ocr_tables(raw) if t.get("html")]


def _table_text(t):
    """Una tabla como texto tabulado (cabecera + filas)."""
    columns = t.get("columns") or (t.get("evaluation") or {}).get("columns")
    rows = t.get("rows")
    if columns and rows:
        lines = ["\t".join(str(c) for c in columns)]
        lines += ["\t".join("" if c is None else str(c) for c in row) for row in rows]
        return "\n".join(lines) + "\n"
    if t.get("html") and str(t["html"]).strip():
        try:
            return "\n".join(df.to_csv(index=False, sep="\t") for df in pd.read_html(StringIO(t["html"])))
        except Exception:
            return ""
    return ""


def ocr_to_tsv(raw, budget=CHAR_BUDGET["tables"]):
    """Todas las tablas de un JSON de OCR como texto tabulado, truncado a `budget` caracteres."""
    out = []
    for t in _iter_ocr_tables(raw):
        text = _table_text(t)
        if text.strip():
            out.append(f"[Table {len(out) + 1}]\n{text}")
    return "\n".join(out)[:budget]


def tables_to_tsv(html_tables, budget=CHAR_BUDGET["tables"]):
    """Compatibilidad: lista de fragmentos HTML -> texto tabulado."""
    return ocr_to_tsv({"results": [{"tables": [{"html": h} for h in html_tables or []]}]}, budget)


# ---------------------------------------------------------------------------
# Entradas por campo
# ---------------------------------------------------------------------------
def _sentences_with_urls(text, budget):
    """Frases del texto completo que contienen URLs o menciones al código."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    hits = [s.strip() for s in sentences if _URL_HINT.search(s)]
    seen, kept = set(), []
    for s in hits:
        if s not in seen:
            seen.add(s)
            kept.append(s)
    return "\n".join(kept)[:budget]


def build_field_inputs(paper):
    """
    Devuelve un dict campo -> texto de entrada. Mismo texto para todas las rutas.
      name      : título + abstract + inicio del texto (introducción)
      tasks     : título + abstract
      category  : título + abstract
      repo      : abstract + frases del paper que contienen URLs o mencionan el código
      datasets  : secciones experimentales + tablas
      metrics   : secciones experimentales + tablas
    """
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    full_text = paper.get("full_text") or ""
    sections = paper.get("sections") or ""
    tables_tsv = paper.get("tables_tsv") or ""

    head = f"Title: {title}\n\nAbstract: {abstract}"
    name_ctx = f"{head}\n\nBeginning of the paper:\n{full_text}"[: CHAR_BUDGET["name"]]

    url_ctx = _sentences_with_urls(full_text, CHAR_BUDGET["repo"])
    repo_ctx = f"Abstract: {abstract}\n\nSentences mentioning links or code:\n{url_ctx or '(none found)'}"

    if not sections:
        # Sin secciones reconocibles: usamos los dos últimos tercios del cuerpo,
        # que es donde suelen estar los experimentos.
        sections = full_text[len(full_text) // 3:]
    exp_ctx = f"Experimental sections:\n{sections[: CHAR_BUDGET['experiments']]}"
    if tables_tsv:
        exp_ctx += f"\n\nTables:\n{tables_tsv}"

    return {
        "name": name_ctx,
        "tasks": head,
        "category": head,
        "repo": repo_ctx,
        "datasets": exp_ctx,
        "metrics": exp_ctx,
    }


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [str(x).strip() for x in v if x is not None and str(x).strip()]


def ground_truth_of(raw_paper, category_field="model_category"):
    """
    GT de un paper. El nombre puede tener alias (build_dataset.py guarda todas las
    variantes que aparecen en los leaderboards); se acierta si coincide con cualquiera.
    """
    name = raw_paper.get("model_name")
    names = [name] if isinstance(name, str) else list(name or [])
    names += [a for a in (raw_paper.get("model_name_aliases") or []) if a not in names]
    names = [n for n in names if n]
    return {
        "name": names if len(names) > 1 else (names[0] if names else None),
        "category": raw_paper.get(category_field),
        "tasks": _as_list(raw_paper.get("tasks")),
        "repo": _as_list(raw_paper.get("paper_repo")),
        "datasets": _as_list(raw_paper.get("Datasets")),
        "metrics": _as_list(raw_paper.get("Metrics")),
    }


def has_rich_gt(gt):
    """Papers con GT de nombre/categoría/datasets/métricas (los únicos que evalúan esos campos)."""
    return bool(gt.get("name") or gt.get("category") or gt.get("datasets") or gt.get("metrics"))


# ---------------------------------------------------------------------------
# Carga del corpus
# ---------------------------------------------------------------------------
def _find_ocr_dir(data_dir):
    for name in ("lightocr_jsons", "lightocr_json"):
        if (data_dir / name).is_dir():
            return data_dir / name
    return None


def load_corpus(data_dir="testing_data", category_field="model_category", verbose=True):
    data_dir = Path(data_dir)
    pwc_path = data_dir / "pwc_final.json"
    if not pwc_path.is_file():
        raise SystemExit(
            f"No existe {pwc_path}: este perfil de dataset todavía no tiene corpus.\n"
            "Para el perfil 'hetero' hay que construirlo antes, en este orden:\n"
            "  python build_dataset.py select\n"
            "  python build_dataset.py pdfs\n"
            "  python build_dataset.py grobid\n"
            "(el perfil 'kge' usa testing_data/, que ya viene en el repo). Ver README_EXPERIMENTS.md.")
    with open(pwc_path, encoding="utf-8") as f:
        raw = json.load(f)

    ocr_dir = _find_ocr_dir(data_dir)
    corpus, skipped, with_tables, seen = [], 0, 0, set()

    for paper in raw:
        xml_path_str = paper.get("local_xml_path")
        if not xml_path_str:
            skipped += 1
            continue
        filename = xml_path_str.replace("\\", "/").split("/")[-1]
        arxiv_id = filename.replace(".tei.xml", "")
        if arxiv_id in seen:
            skipped += 1
            continue
        xml_path = data_dir / "xml_outputs" / filename
        try:
            parser = XMLParser(xml_path)
            full_text = parser.get_full_text() or ""
            abstract = parser.get_abstract() or paper.get("abstract") or ""
            # Pedimos todas las secciones y filtramos aquí: get_sections(target)
            # sobrescribe secciones distintas que coinciden con el mismo objetivo.
            all_sec = parser.get_sections(target_sections=None) or {}
            sections = "\n\n".join(
                v for k, v in all_sec.items()
                if v and any(t in k.lower() for t in EXPERIMENT_SECTIONS)
            )
        except Exception:
            skipped += 1
            continue

        tables_tsv, has_ocr = "", False
        if ocr_dir is not None:
            ocr_path = ocr_dir / f"{arxiv_id}_lightonocr.json"
            if ocr_path.is_file():
                try:
                    with open(ocr_path, encoding="utf-8") as f:
                        tables_tsv = ocr_to_tsv(json.load(f))
                    has_ocr = True            # el OCR se hizo (aunque no encontrara ninguna tabla)
                except Exception:
                    tables_tsv = ""
        with_tables += bool(tables_tsv)

        seen.add(arxiv_id)
        corpus.append({
            "arxiv_id": arxiv_id,
            "title": paper.get("title") or "",
            "authors": paper.get("authors") or [],
            "abstract": abstract,
            "full_text": full_text,
            "sections": sections,
            "tables_tsv": tables_tsv,
            "has_ocr": has_ocr,
            "gt": ground_truth_of(paper, category_field),
            "area": paper.get("area"),
            "split": paper.get("split"),
        })

    if not corpus:
        n_xml = sum(1 for p in raw if p.get("local_xml_path"))
        raise SystemExit(
            f"{pwc_path} tiene {len(raw)} papers, pero ninguno se ha podido cargar "
            f"({n_xml} con 'local_xml_path'; los XML se buscan en {data_dir / 'xml_outputs'}/).\n"
            "Faltan los PDFs y/o su conversión a XML. Ejecuta, en este orden:\n"
            "  python build_dataset.py pdfs\n"
            "  python build_dataset.py grobid\n"
            "y comprueba el resultado con: python build_dataset.py stats")
    if verbose:
        print(f"[corpus] {len(corpus)} papers cargados, {skipped} descartados, {with_tables} con tablas OCR.")
        if skipped > 0.1 * len(raw):
            print(f"[corpus] AVISO: se han descartado {skipped} de {len(raw)} papers (sin XML o con XML ilegible). "
                  "Revisa las fases 'pdfs' y 'grobid' con: python build_dataset.py stats")
    return corpus


# ---------------------------------------------------------------------------
# Partición fija test / pool
# ---------------------------------------------------------------------------
def make_split(corpus, n_test=400, seed=13):
    """
    Si todos los papers traen el campo "split" ("test"/"pool"), se respeta tal cual.
    En caso contrario:
    Test = TODOS los papers con GT rico (nombre/categoría/datasets/métricas) + una muestra
    aleatoria del resto hasta n_test. El pool (resto) es de donde salen los casos.
    Como los casos se construyen con predicciones del LLM (no con GT), meter los
    papers con GT rico en test no causa fuga de información.
    """
    # Partición predefinida (build_dataset.py la fija estratificada por área).
    if corpus and all(p.get("split") in ("test", "pool") for p in corpus):
        return {
            "seed": "predefined",
            "test": sorted(p["arxiv_id"] for p in corpus if p["split"] == "test"),
            "pool": sorted(p["arxiv_id"] for p in corpus if p["split"] == "pool"),
        }
    rng = random.Random(seed)
    rich = [p["arxiv_id"] for p in corpus if has_rich_gt(p["gt"])]
    others = [p["arxiv_id"] for p in corpus if not has_rich_gt(p["gt"])]
    rng.shuffle(others)
    n_extra = max(0, n_test - len(rich))
    test = sorted(rich + others[:n_extra])
    pool = sorted(others[n_extra:])
    return {"seed": seed, "test": test, "pool": pool}
