"""
Construye un corpus HETEROGÉNEO y COMPLETO para P2MC-NeSy a partir de pwc-archive
(volcado final de Papers with Code, julio de 2025).

Basado en GAP-KGE/pwc_extraction.py, con estos bugs corregidos:
  * Cruce de papers por ID de arXiv normalizado (antes: URL exacta, de modo que
    http/https o v1/v3 impedían el cruce y se perdían la mayoría de papers).
  * Métricas acumuladas sobre TODAS las filas del leaderboard (antes se reiniciaban
    en cada fila y solo sobrevivían las de la última).
  * Datasets realmente guardados (la línea estaba comentada).
  * Recorrido recursivo de subtareas y subdatasets.
  * Nombre del modelo: el más frecuente en los leaderboards, con el resto como alias.
  * Rutas de PDF/XML persistidas en el JSON tras cada fase (antes se perdían).

Partición:
  test = papers COMPLETOS (nombre, tarea, dataset, métrica, repositorio y área),
         estratificados por área de investigación.
  pool = papers de las mismas áreas sin exigir completitud: solo alimentan la base
         de casos, que se construye con predicciones del LLM, no con el GT.

Fases (todas reanudables):
  python build_dataset.py select              # GT + partición -> testing_data_hetero/pwc_final.json
  python build_dataset.py pdfs                # descarga de PDFs de arXiv (3 s entre peticiones)
  python build_dataset.py grobid              # PDF -> TEI XML (GROBID en localhost:8070)
  python build_dataset.py ocr                 # opcional: tablas con LightOnOCR
  python build_dataset.py stats               # resumen del corpus final
  python build_dataset.py all                 # select + pdfs + grobid + stats

Requiere: pip install datasets requests (y GROBID en marcha para la fase grobid).
"""
import argparse
import fcntl
import gzip
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

HF_SOURCES = {
    "papers": "pwc-archive/papers-with-abstracts",
    "links": "pwc-archive/links-between-paper-and-code",
    "evals": "pwc-archive/evaluation-tables",
}

# Áreas de PwC demasiado genéricas para estratificar.
DEFAULT_EXCLUDED_AREAS = {"Methodology", "Miscellaneous"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
_NEW_ID = re.compile(r"^(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?:v\d+)?$")                      # YYMM.NNNN(N)
_OLD_ID = re.compile(r"^([a-z]+(?:-[a-z]+)*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?$", re.IGNORECASE)  # cs/0101001
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html|format)/", re.IGNORECASE)


def arxiv_id_from(value):
    """
    ID de arXiv sin versión, o None.
    Solo se acepta (a) un ID tal cual ("1706.03762v5", "cs/0101001") o (b) una URL de
    arxiv.org. Antes se buscaba el patrón en CUALQUIER URL, y se inventaban IDs a partir
    de enlaces de IEEE ("document/9555146"), ACM (el DOI 10.1145/3583780.3615067 daba
    "3780.36150") o ScienceDirect, con lo que esos papers se intentaban bajar de arXiv.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    m = _ARXIV_URL.search(v)
    if m:
        v = v[m.end():]
    elif re.match(r"^[a-z]+://", v, re.IGNORECASE):
        return None  # una URL que no es de arXiv
    v = re.sub(r"^arxiv:", "", v, flags=re.IGNORECASE)
    v = re.split(r"[?#]", v)[0].rstrip("/")
    v = re.sub(r"\.pdf$", "", v, flags=re.IGNORECASE)
    for pat in (_NEW_ID, _OLD_ID):
        mm = pat.match(v)
        if mm:
            return mm.group(1).lower()
    return None


def norm_url(url):
    if not url or not isinstance(url, str):
        return ""
    u = url.strip().lower()
    u = re.sub(r"^[a-z]+://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = re.sub(r"v\d+(\.pdf)?$", "", u)
    return u.rstrip("/")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def safe_id(arxiv_id):
    """IDs antiguos tipo cs/0101001 -> cs_0101001 (válido como nombre de fichero)."""
    return arxiv_id.replace("/", "_")


def as_obj(value):
    """Los campos anidados pueden venir como JSON en texto según el formato del volcado."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def as_list(value):
    value = as_obj(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


def as_dict(value):
    value = as_obj(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):  # [(clave, valor), ...] o [{"key":..,"value":..}]
        out = {}
        for item in value:
            if isinstance(item, dict) and "key" in item:
                out[item["key"]] = item.get("value")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out[item[0]] = item[1]
        return out
    return {}


def _has_value(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    return bool(s) and s not in ("none", "nan", "-", "null")


# ---------------------------------------------------------------------------
# Carga de fuentes
# ---------------------------------------------------------------------------
def load_source(key, local_dir=None):
    """Hugging Face por defecto; --local-dir permite usar ficheros descargados (json, json.gz, jsonl, parquet)."""
    if local_dir:
        base = Path(local_dir)
        stem = HF_SOURCES[key].split("/")[-1]
        for ext in (".json", ".json.gz", ".jsonl", ".parquet"):
            path = base / f"{stem}{ext}"
            if not path.is_file():
                continue
            if ext == ".parquet":
                import pandas as pd
                return pd.read_parquet(path).to_dict("records")
            opener = gzip.open if ext.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                if ext == ".jsonl":
                    return [json.loads(line) for line in f if line.strip()]
                return json.load(f)
        raise FileNotFoundError(f"No encuentro {stem}.(json|json.gz|jsonl|parquet) en {base}")
    from datasets import load_dataset
    t0 = time.time()
    print(f"[select] descargando/cargando {HF_SOURCES[key]} (la 1ª vez baja de Hugging Face; luego va por caché) ...", flush=True)
    ds = load_dataset(HF_SOURCES[key], split="train")
    print(f"[select]   {HF_SOURCES[key]}: {len(ds)} filas en {time.time() - t0:.0f}s", flush=True)
    return ds


# ---------------------------------------------------------------------------
# Construcción del ground truth
# ---------------------------------------------------------------------------
def walk_tasks(task_row, inherited_areas=()):
    """Recorre una tarea y sus subtareas -> (nombre, áreas, sinónimos, lista de datasets)."""
    name = task_row.get("task") or task_row.get("name") or ""
    areas = [a for a in as_list(task_row.get("categories")) if a] or list(inherited_areas)
    synonyms = [s for s in as_list(task_row.get("synonyms")) if s]
    yield name, areas, synonyms, as_list(task_row.get("datasets"))
    for sub in as_list(task_row.get("subtasks")):
        if isinstance(sub, dict):
            yield from walk_tasks(sub, areas)


def walk_datasets(ds_row):
    """Recorre un dataset y sus subdatasets -> (nombre, filas sota)."""
    name = ds_row.get("dataset") or ds_row.get("subdataset") or ds_row.get("name") or ""
    sota = as_obj(ds_row.get("sota")) or {}
    rows = as_list(sota.get("rows")) if isinstance(sota, dict) else []
    yield name, rows
    for sub in as_list(ds_row.get("subdatasets")):
        if isinstance(sub, dict):
            yield from walk_datasets(sub)


def build_ground_truth(papers_src, links_src, evals_src, repo_policy="official_or_mentioned"):
    # Índice de papers por ID de arXiv y por URL normalizada
    t0 = time.time()
    papers, by_url = {}, {}
    for p in tqdm(papers_src, desc="[select] papers", unit="paper", mininterval=10, leave=True):
        p = dict(p)
        aid = arxiv_id_from(p.get("arxiv_id")) or arxiv_id_from(p.get("url_abs"))
        if not aid:
            continue
        p["arxiv_id"] = aid
        papers[aid] = p
        for u in (p.get("url_abs"), p.get("url_pdf"), p.get("paper_url")):
            if u:
                by_url[norm_url(u)] = aid
    print(f"[select] {len(papers)} papers con ID de arXiv ({time.time() - t0:.0f}s)", flush=True)

    def resolve(url):
        return arxiv_id_from(url) or by_url.get(norm_url(url))

    # Repositorios
    t0 = time.time()
    repos, all_repos = defaultdict(list), defaultdict(list)
    for l in tqdm(links_src, desc="[select] enlaces a código", unit="enlace", mininterval=10):
        aid = arxiv_id_from(l.get("paper_arxiv_id")) or resolve(l.get("paper_url_abs")) or resolve(l.get("paper_url"))
        repo = l.get("repo_url")
        if not aid or not repo:
            continue
        all_repos[aid].append(repo)
        official = bool(l.get("is_official"))
        mentioned = bool(l.get("mentioned_in_paper"))
        keep = official if repo_policy == "official" else (official or mentioned)
        if keep and repo not in repos[aid]:
            repos[aid].append(repo)

    print(f"[select] enlaces procesados ({time.time() - t0:.0f}s)", flush=True)

    # Tablas de evaluación
    t0 = time.time()
    gt = defaultdict(lambda: {"datasets": set(), "metrics": set(), "names": Counter(),
                              "eval_tasks": set(), "areas": Counter()})
    task_vocab = {}
    unresolved = 0
    for row in tqdm(evals_src, desc="[select] tablas de evaluación (tareas)", unit="tarea", mininterval=10):
        for t_name, t_areas, t_syn, t_datasets in walk_tasks(dict(row)):
            if t_name:
                entry = task_vocab.setdefault(t_name, {"areas": set(), "synonyms": set()})
                entry["areas"].update(t_areas)
                entry["synonyms"].update(t_syn)
            for ds in t_datasets:
                ds = as_obj(ds)
                if not isinstance(ds, dict):
                    continue
                for ds_name, rows in walk_datasets(ds):
                    for r in rows:
                        r = as_obj(r)
                        if not isinstance(r, dict):
                            continue
                        aid = resolve(r.get("paper_url") or "")
                        if not aid:
                            unresolved += 1
                            continue
                        g = gt[aid]
                        metrics = {k for k, v in as_dict(r.get("metrics")).items() if k and _has_value(v)}
                        if not metrics:
                            continue  # fila sin resultados: no aporta dataset ni métrica
                        g["metrics"].update(metrics)
                        if ds_name:
                            g["datasets"].add(ds_name)
                        if r.get("model_name"):
                            g["names"][str(r["model_name"]).strip()] += 1
                        if t_name:
                            g["eval_tasks"].add(t_name)
                        for a in t_areas:
                            g["areas"][a] += 1
    print(f"[select] {len(gt)} papers con resultados en leaderboards "
          f"({unresolved} filas sin paper identificable) ({time.time() - t0:.0f}s)", flush=True)
    return papers, repos, all_repos, gt, task_vocab


def primary_area(area_counter, allowed):
    ranked = sorted(((n, a) for a, n in area_counter.items() if a in allowed), key=lambda x: (-x[0], x[1]))
    return ranked[0][1] if ranked else None


def make_record(p, aid, g, repos, all_repos, task_areas, allowed, split):
    tasks = sorted(set(as_list(p.get("tasks"))) | set(g["eval_tasks"] if g else ()))
    if g and g["areas"]:
        area_counter = g["areas"]
    else:  # pool: área a partir de las tareas del paper
        area_counter = Counter(a for t in tasks for a in task_areas.get(t, ()))
    names = [n for n, _ in g["names"].most_common()] if g else []
    return {
        "paper_url": p.get("paper_url"),
        "arxiv_id": aid,
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "short_abstract": p.get("short_abstract"),
        "url_abs": p.get("url_abs") or f"https://arxiv.org/abs/{aid}",
        "url_pdf": p.get("url_pdf") or f"https://arxiv.org/pdf/{aid}",
        "proceeding": p.get("proceeding"),
        "date": str(p.get("date") or ""),
        "authors": [a for a in as_list(p.get("authors")) if a],
        "tasks": tasks,
        "methods": [m.get("name") if isinstance(m, dict) else m for m in as_list(p.get("methods"))],
        "model_category": None,
        "Datasets": sorted(g["datasets"]) if g else [],
        "Metrics": sorted(g["metrics"]) if g else [],
        "model_name": names[0] if names else None,
        "model_name_aliases": names[1:10],
        "paper_repo": list(repos.get(aid, [])),
        "repo_url": sorted(set(all_repos.get(aid, []))),
        "area": primary_area(area_counter, allowed),
        "areas": sorted(a for a in area_counter if a),
        "split": split,
        "local_pdf_path": None,
        "local_xml_path": None,
    }


def is_complete(rec):
    return all([rec["arxiv_id"], rec["title"], rec["abstract"], rec["model_name"], rec["tasks"],
                rec["Datasets"], rec["Metrics"], rec["paper_repo"], rec["area"]])


def _year(p):
    m = re.match(r"(\d{4})", str(p.get("date") or ""))
    return int(m.group(1)) if m else None


def select(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_src = load_source("papers", args.local_dir)
    links_src = load_source("links", args.local_dir)
    evals_src = load_source("evals", args.local_dir)
    papers, repos, all_repos, gt, task_vocab = build_ground_truth(
        papers_src, links_src, evals_src, repo_policy=args.repo_policy)
    task_areas = {t: v["areas"] for t, v in task_vocab.items()}

    def in_years(p):
        y = _year(p)
        return (args.min_year is None or (y and y >= args.min_year)) and \
               (args.max_year is None or (y and y <= args.max_year))

    excluded = set(args.exclude_areas if args.exclude_areas is not None else DEFAULT_EXCLUDED_AREAS)

    # Candidatos completos (test). El área primaria se calcula ignorando las áreas excluidas.
    candidates = []
    for aid, g in gt.items():
        p = papers.get(aid)
        if not p or not in_years(p):
            continue
        allowed_here = set(g["areas"]) - excluded
        rec = make_record(p, aid, g, repos, all_repos, task_areas, allowed_here, "test")
        if is_complete(rec):
            candidates.append((rec, g))
    all_areas = Counter(rec["area"] for rec, _ in candidates)
    print("[select] papers completos por área:", dict(all_areas.most_common(20)))

    if args.areas:
        target = list(args.areas)
    else:
        target = [a for a, n in all_areas.most_common() if n >= args.n_test_per_area][: args.n_areas]
    print(f"[select] áreas elegidas: {target}")
    allowed = set(target)

    # Área primaria restringida a las áreas elegidas (un paper CV+Methodology cuenta como CV)
    complete = []
    for rec, g in candidates:
        rec["area"] = primary_area(g["areas"], allowed)
        if rec["area"]:
            complete.append(rec)

    rng = random.Random(args.seed)
    test, test_ids = [], set()
    for area in target:
        cands = sorted((r for r in complete if r["area"] == area), key=lambda r: r["arxiv_id"])
        rng.shuffle(cands)
        chosen = cands[: args.n_test_per_area]
        if len(chosen) < args.n_test_per_area:
            print(f"[select] AVISO: {area} solo tiene {len(chosen)} papers completos")
        test.extend(chosen)
        test_ids.update(r["arxiv_id"] for r in chosen)

    # Pool: papers de las mismas áreas, sin exigir completitud
    pool_by_area = defaultdict(list)
    for aid, p in papers.items():
        if aid in test_ids or not p.get("abstract") or not in_years(p):
            continue
        rec = make_record(p, aid, gt.get(aid), repos, all_repos, task_areas, allowed, "pool")
        if rec["area"] in allowed:
            pool_by_area[rec["area"]].append(rec)
    pool = []
    for area in target:
        cands = sorted(pool_by_area[area], key=lambda r: r["arxiv_id"])
        rng.shuffle(cands)
        pool.extend(cands[: args.n_pool_per_area])

    corpus = test + pool
    _atomic_write_json(out_dir / "pwc_final.json", corpus)
    with open(out_dir / "areas.json", "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=1)

    # Vocabulario de tareas: tablas de evaluación + tareas con al menos N papers
    task_counts = Counter(t for p in papers.values() for t in as_list(p.get("tasks")))
    vocab_names = set(task_vocab) | {t for t, n in task_counts.items() if n >= args.min_task_papers}
    vocab = [{
        "uri": f"https://linkedpaperswithcode.com/task/{slug(t)}",
        "name": t,
        "areas": [f"https://linkedpaperswithcode.com/area/{slug(a)}" for a in sorted(task_vocab.get(t, {}).get("areas", []))],
        "synonyms": sorted(task_vocab.get(t, {}).get("synonyms", [])),
        "numPapers": task_counts.get(t, 0),
    } for t in sorted(vocab_names)]
    with open(out_dir / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)

    print(f"[select] test={len(test)} pool={len(pool)} vocabulario de tareas={len(vocab)} -> {out_dir}/")
    stats(args)


# ---------------------------------------------------------------------------
# PDFs, GROBID y OCR (reanudables; el JSON se guarda tras cada fase)
# ---------------------------------------------------------------------------
def _load_corpus(out_dir):
    with open(Path(out_dir) / "pwc_final.json", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path, obj, indent=1):
    """
    Escribe JSON de forma atómica. El fichero temporal lleva el PID del proceso:
    con un nombre fijo, dos procesos se pisaban el .tmp y el segundo `replace`
    fallaba con FileNotFoundError.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # solo si algo falló antes del replace
            tmp.unlink()


def _save_corpus(out_dir, corpus):
    _atomic_write_json(Path(out_dir) / "pwc_final.json", corpus)


_LOCK_HANDLE = None


def acquire_lock(out_dir):
    """
    Impide dos build_dataset.py a la vez sobre la misma carpeta: se sobrescribirían
    el progreso (cada uno guarda su copia del corpus en memoria).
    """
    global _LOCK_HANDLE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / ".build_dataset.lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        pid = fh.read().strip() or "desconocido"
        raise SystemExit(
            f"Ya hay otro build_dataset.py trabajando sobre {out_dir}/ (PID {pid}).\n"
            f"Espera a que termine o páralo con: kill {pid}\n"
            "Para ver los que hay en marcha: pgrep -fa build_dataset")
    except OSError as e:  # p. ej. sistemas de ficheros sin soporte de flock
        print(f"[lock] AVISO: no se pudo bloquear {out_dir}/ ({e}). Asegúrate de no lanzar dos instancias a la vez.")
        return
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _LOCK_HANDLE = fh


def download_pdfs(args):
    import requests
    out_dir = Path(args.out_dir)
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus(out_dir)
    headers = {"User-Agent": "p2mc-nesy-dataset-builder (research; mailto:%s)" % args.contact}
    ok = fail = 0
    for n, rec in enumerate(corpus, 1):
        path = pdf_dir / f"{safe_id(rec['arxiv_id'])}.pdf"
        if path.is_file() and path.stat().st_size > 10_000:
            rec["local_pdf_path"] = str(path)
            continue
        url = f"https://arxiv.org/pdf/{rec['arxiv_id']}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    path.write_bytes(r.content)
                    rec["local_pdf_path"] = str(path)
                    ok += 1
                    break
                if r.status_code in (429, 503):
                    time.sleep(30 * (attempt + 1))
                    continue
                break
            except Exception:
                time.sleep(10)
        if not rec.get("local_pdf_path"):
            fail += 1
        time.sleep(args.delay)  # arXiv pide no superar 1 petición cada ~3 s
        if n % 50 == 0:
            _save_corpus(out_dir, corpus)
            print(f"[pdfs] {n}/{len(corpus)}  nuevos={ok} fallos={fail}")
    _save_corpus(out_dir, corpus)
    print(f"[pdfs] descargados={ok} fallos={fail}")


def run_grobid(args):
    import requests
    out_dir = Path(args.out_dir)
    xml_dir = out_dir / "xml_outputs"
    xml_dir.mkdir(parents=True, exist_ok=True)
    try:
        requests.get(f"{args.grobid_url}/api/isalive", timeout=10).raise_for_status()
    except Exception as e:
        raise SystemExit(f"GROBID no responde en {args.grobid_url}: {e}")
    corpus = _load_corpus(out_dir)

    def work(rec):
        pdf = rec.get("local_pdf_path")
        xml = xml_dir / f"{safe_id(rec['arxiv_id'])}.tei.xml"
        if xml.is_file():
            return rec, str(xml)
        if not pdf or not Path(pdf).is_file():
            return rec, None
        try:
            with open(pdf, "rb") as f:
                r = requests.post(f"{args.grobid_url}/api/processFulltextDocument",
                                  files={"input": f},
                                  data={"generateTeiIds": "1", "consolidateHeader": "1",
                                        "consolidateConclusion": "1"},
                                  timeout=300)
            if r.status_code == 200:
                xml.write_text(r.text, encoding="utf-8")
                return rec, str(xml)
        except Exception as e:
            print(f"[grobid] {rec['arxiv_id']}: {e}")
        return rec, None

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, r) for r in corpus]):
            rec, xml = fut.result()
            rec["local_xml_path"] = xml
            done += 1
            if done % 100 == 0:
                _save_corpus(out_dir, corpus)
                print(f"[grobid] {done}/{len(corpus)}")
    _save_corpus(out_dir, corpus)
    print(f"[grobid] con XML: {sum(1 for r in corpus if r.get('local_xml_path'))}/{len(corpus)}")


def run_ocr(args):
    """Tablas con LightOnOCR (lento: GPU). Por defecto solo para el test."""
    from parsers.lightocr_parser import LightOcrParser
    out_dir = Path(args.out_dir)
    ocr_dir = out_dir / "lightocr_jsons"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus(out_dir)
    targets = [r for r in corpus if r.get("local_pdf_path") and (args.ocr_pool or r["split"] == "test")]
    parser = LightOcrParser()
    for n, rec in enumerate(targets, 1):
        out = ocr_dir / f"{safe_id(rec['arxiv_id'])}_lightonocr.json"
        if not out.is_file():
            try:
                parser.process(rec["local_pdf_path"], str(out))
            except Exception as e:
                print(f"[ocr] {rec['arxiv_id']}: {e}")
        if n % 25 == 0:
            print(f"[ocr] {n}/{len(targets)}")


def _tei_title(xml_path):
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(Path(xml_path).read_text(encoding="utf-8"), "xml")
        node = soup.find("titleStmt")
        node = node.find("title") if node else None
        return node.get_text(" ", strip=True) if node else ""
    except Exception:
        return ""


def prune(args):
    """
    Deja en el corpus solo papers con PDF y, si ya hay XML (--check-titles), cuyo título
    en el XML coincida con el del ground truth: así un PDF equivocado (p. ej. de un ID de
    arXiv inventado que coincide con otro paper) nunca se evalúa contra el GT de otro.
    Guarda una copia en pwc_final.before_prune.json.
    """
    from rapidfuzz import fuzz
    out_dir = Path(args.out_dir)
    corpus = _load_corpus(out_dir)
    pdf_dir, xml_dir = out_dir / "pdfs", out_dir / "xml_outputs"
    keep, dropped, unverifiable = [], defaultdict(list), 0

    if args.check_titles:
        with_pdf = [r for r in corpus if (pdf_dir / f"{safe_id(r['arxiv_id'])}.pdf").is_file()]
        with_xml = sum(1 for r in with_pdf if (xml_dir / f"{safe_id(r['arxiv_id'])}.tei.xml").is_file())
        if with_pdf and with_xml < 0.9 * len(with_pdf):
            raise SystemExit(
                f"Solo hay XML para {with_xml} de {len(with_pdf)} PDFs: --check-titles descartaría casi todo.\n"
                "Termina primero la fase 'grobid' (python build_dataset.py grobid) y comprueba con 'stats'.")

    def norm(t):
        return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()

    for r in corpus:
        sid = safe_id(r["arxiv_id"])
        if not (pdf_dir / f"{sid}.pdf").is_file():
            dropped["sin PDF"].append(r)
            continue
        if args.check_titles:
            xml = xml_dir / f"{sid}.tei.xml"
            if not xml.is_file():
                dropped["sin XML"].append(r)
                continue
            title = _tei_title(xml)
            if not title:
                unverifiable += 1          # GROBID no sacó título: no se puede comprobar, se conserva
            elif fuzz.token_set_ratio(norm(title), norm(r.get("title"))) < args.title_threshold:
                dropped["el título del PDF no coincide con el del GT"].append(r)
                continue
        keep.append(r)

    total_dropped = len(corpus) - len(keep)
    print(f"[prune] {len(corpus)} papers -> {len(keep)} conservados, {total_dropped} descartados"
          + (f" ({unverifiable} sin título verificable en el XML)" if args.check_titles else ""))
    for reason, rows in dropped.items():
        c = Counter(r["split"] for r in rows)
        print(f"[prune]   {reason}: {len(rows)} (test={c['test']}, pool={c['pool']})")
        for r in rows[:5]:
            print(f"[prune]      {r['arxiv_id']}  {str(r.get('title'))[:70]}")
    if total_dropped:
        backup = out_dir / "pwc_final.before_prune.json"
        if not backup.exists():
            _atomic_write_json(backup, corpus)
        _save_corpus(out_dir, keep)
        print(f"[prune] copia de seguridad en {backup}")
    stats(args)


def stats(args):
    corpus = _load_corpus(args.out_dir)
    by = Counter((r["split"], r["area"]) for r in corpus)
    with_xml = Counter((r["split"], r["area"]) for r in corpus if r.get("local_xml_path"))
    areas = sorted({a for _, a in by})
    report = {"areas": {}, "totals": {}}
    print(f"{'área':40s} {'test':>6s} {'pool':>6s} {'test_xml':>9s} {'pool_xml':>9s}")
    for a in areas:
        row = {s: by[(s, a)] for s in ("test", "pool")}
        row.update({f"{s}_xml": with_xml[(s, a)] for s in ("test", "pool")})
        report["areas"][a] = row
        print(f"{a:40s} {row['test']:6d} {row['pool']:6d} {row['test_xml']:9d} {row['pool_xml']:9d}")
    test = [r for r in corpus if r["split"] == "test"]
    report["totals"] = {
        "test": len(test), "pool": len(corpus) - len(test),
        "test_mean_datasets": round(sum(len(r["Datasets"]) for r in test) / max(1, len(test)), 2),
        "test_mean_metrics": round(sum(len(r["Metrics"]) for r in test) / max(1, len(test)), 2),
        "test_mean_tasks": round(sum(len(r["tasks"]) for r in test) / max(1, len(test)), 2),
        "test_with_name_aliases": sum(1 for r in test if r["model_name_aliases"]),
    }
    print(report["totals"])
    with open(Path(args.out_dir) / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["select", "pdfs", "grobid", "ocr", "prune", "stats", "all"])
    ap.add_argument("--out-dir", default="testing_data_hetero")
    ap.add_argument("--local-dir", default=None, help="usar volcados locales en vez de Hugging Face")
    ap.add_argument("--n-areas", type=int, default=8)
    ap.add_argument("--areas", nargs="*", help="lista explícita de áreas (si no, las n más pobladas)")
    ap.add_argument("--exclude-areas", nargs="*", default=None)
    ap.add_argument("--n-test-per-area", type=int, default=75)
    ap.add_argument("--n-pool-per-area", type=int, default=300)
    ap.add_argument("--min-year", type=int, default=None)
    ap.add_argument("--max-year", type=int, default=None)
    ap.add_argument("--min-task-papers", type=int, default=5, help="mínimo de papers para que una tarea entre en el vocabulario")
    ap.add_argument("--repo-policy", choices=["official", "official_or_mentioned"], default="official_or_mentioned",
                    help="qué enlaces de código cuentan como repositorio del paper")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--delay", type=float, default=3.0, help="segundos entre descargas de arXiv")
    ap.add_argument("--contact", default="p2mc@oeg.upm.es", help="email de contacto en el User-Agent")
    ap.add_argument("--grobid-url", default="http://localhost:8070")
    ap.add_argument("--workers", type=int, default=4, help="peticiones simultáneas a GROBID")
    ap.add_argument("--ocr-pool", action="store_true", help="aplicar OCR también al pool")
    ap.add_argument("--check-titles", action="store_true",
                    help="prune: además, descarta papers cuyo título en el XML no coincide con el del GT (tras 'grobid')")
    ap.add_argument("--title-threshold", type=int, default=70, help="prune: similitud mínima de títulos (0-100)")
    args = ap.parse_args()

    if args.phase != "stats":
        acquire_lock(args.out_dir)

    if args.phase in ("select", "all"):
        select(args)
    if args.phase in ("pdfs", "all"):
        download_pdfs(args)
    if args.phase in ("grobid", "all"):
        run_grobid(args)
    if args.phase == "ocr":
        run_ocr(args)
    if args.phase == "prune":
        prune(args)
    if args.phase in ("stats", "all") and args.phase != "select":
        stats(args)


if __name__ == "__main__":
    main()
