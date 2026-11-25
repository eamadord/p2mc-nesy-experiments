"""
Experimento P2MC-NeSy: ¿mantiene el CBR (modelo pequeño + casos) la calidad de
la extracción con LLMs grandes, reduciendo tiempo y recursos?

Diseño:
  * Test FIJO (todos los papers con GT rico + muestra aleatoria), evaluado con
    TODAS las rutas -> comparaciones pareadas sobre los mismos papers.
  * La ruta LLM se ejecuta UNA vez sobre test + pool y se cachea. La base de
    casos de umbral N = primeras N predicciones LLM del pool (ordenación
    aleatoria por semilla). Así cada umbral solo ejecuta el modelo pequeño.
  * Ablaciones:
      small            modelo pequeño sin casos         -> ¿aporta algo el caso?
      retrieval_only   copiar el caso, sin LLM           -> ¿aporta algo el LLM?
      cbr / gt_loo     casos con GT (leave-one-out test) -> techo del método

Uso (desde la raíz del repo):
  python experiment_cbr.py prepare  --dataset hetero
  python experiment_cbr.py llm      --dataset hetero --generation gen2026
  python experiment_cbr.py small    --dataset hetero --generation gen2026
  python experiment_cbr.py cbr      --dataset hetero --generation gen2026
  python experiment_cbr.py oracle   --dataset hetero --generation gen2026
  python experiment_cbr.py all      --dataset hetero --generation gen2026
  python evaluate_predictions.py    --dataset hetero

Añade --mock para probar todo sin GPU (resultados sin significado).
Todas las fases se pueden interrumpir y reanudar.
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from model_card_generation_pipeline import FieldPipeline
from utils.case_base import CaseBase, Embedder
from utils.paper_loader import load_corpus, make_split

# Se fijan en main() según --dataset (perfil de config.DATASETS).
EXP = config.dataset_profile()
OUT = Path(EXP["out_dir"]) / EXP["name"]
CACHE = OUT / "cache"


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _done_ids(path):
    return {r["arxiv_id"] for r in _read_jsonl(path)}


def pred_dir(generation):
    return OUT / generation / "predictions"


# ---------------------------------------------------------------------------
# Fase 0: preparar corpus, partición y embeddings
# ---------------------------------------------------------------------------
def prepare(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    corpus_path = CACHE / "corpus.jsonl"
    if corpus_path.is_file() and not args.force:
        print(f"[prepare] {corpus_path} ya existe (usa --force para regenerarlo).")
        corpus = _read_jsonl(corpus_path)
    else:
        corpus = load_corpus(EXP["data_dir"], category_field=EXP["category_field"])
        corpus_path.unlink(missing_ok=True)
        for p in corpus:
            _append_jsonl(corpus_path, p)

    split_path = OUT / "split.json"
    if not split_path.is_file() or args.force:
        split = make_split(corpus, n_test=EXP["n_test"] or 400, seed=EXP["split_seed"])
        split_path.write_text(json.dumps(split, indent=1))
    split = json.loads(split_path.read_text())
    print(f"[prepare] test={len(split['test'])} pool={len(split['pool'])}")
    by_id = {p["arxiv_id"]: p for p in corpus}
    ocr_test = sum(1 for i in split["test"] if by_id[i].get("has_ocr"))
    ocr_pool = sum(1 for i in split["pool"] if by_id[i].get("has_ocr"))
    print(f"[prepare] con OCR: test={ocr_test}/{len(split['test'])} pool={ocr_pool}/{len(split['pool'])}"
          " (las tablas se guardan AHORA: si el OCR no ha terminado, repite con --force)")
    if max(EXP["thresholds"]) > len(split["pool"]):
        print(f"[prepare] AVISO: el umbral máximo ({max(EXP['thresholds'])}) supera el pool.")

    emb_path = CACHE / f"embeddings_{EXP['embedder']}.npz"
    if not emb_path.is_file() or args.force:
        embedder = Embedder(EXP["embedder"])
        abstracts = [p["abstract"] for p in corpus]
        if EXP["embedder"] == "tfidf":
            embedder.fit(abstracts)
        t0 = time.perf_counter()
        emb = embedder.encode(abstracts)
        print(f"[prepare] embeddings {emb.shape} en {time.perf_counter() - t0:.1f}s")
        np.savez(emb_path, ids=np.array([p["arxiv_id"] for p in corpus]), emb=emb)


POOL_OCR_ONLY = False   # se fija en main() con --pool-ocr-only


def load_state():
    corpus = {p["arxiv_id"]: p for p in _read_jsonl(CACHE / "corpus.jsonl")}
    if not corpus:
        raise SystemExit("Ejecuta primero: python experiment_cbr.py prepare")
    split = json.loads((OUT / "split.json").read_text())
    z = np.load(CACHE / f"embeddings_{EXP['embedder']}.npz")
    emb = {i: e for i, e in zip(z["ids"].tolist(), z["emb"])}
    if POOL_OCR_ONLY:
        # El pool se limita a los papers con OCR hecho, para que los casos de la base se hayan
        # generado con las mismas entradas (tablas incluidas) que el test.
        full = len(split["pool"])
        split = {**split, "pool": [p for p in split["pool"] if corpus[p].get("has_ocr")]}
        sin_ocr = sum(1 for p in split["test"] if not corpus[p].get("has_ocr"))
        print(f"[pool] --pool-ocr-only: {len(split['pool'])} de {full} papers del pool tienen OCR", flush=True)
        if sin_ocr:
            print(f"[pool] AVISO: {sin_ocr} de {len(split['test'])} papers de test no tienen OCR", flush=True)
        if split["pool"] and max(EXP["thresholds"]) > len(split["pool"]):
            print(f"[pool] AVISO: el umbral máximo ({max(EXP['thresholds'])}) supera el pool con OCR; "
                  "habrá menos casos de los previstos", flush=True)
    return corpus, split, emb


# ---------------------------------------------------------------------------
# Utilidades de ejecución
# ---------------------------------------------------------------------------
def make_pipeline(args):
    client = None
    if args.mock:
        from extractors.field_extractor import MockOllamaClient
        client = MockOllamaClient()
    return FieldPipeline(generation=args.generation, client=client, profile=EXP)


def warm_up(extractor, models):
    """Carga cada modelo en VRAM antes de cronometrar nada."""
    for m in sorted(set(models)):
        try:
            extractor.client.chat(model=m, messages=[{"role": "user", "content": "ok"}],
                                  options={"num_predict": 1, "num_ctx": extractor.num_ctx},
                                  keep_alive=extractor.keep_alive)
        except Exception as e:
            print(f"[warm-up] {m}: {e}")


def save_run_meta(generation, name, pipeline, extra=None):
    meta = {
        "generation": generation,
        "run": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "large_models": pipeline.large_models,
        "small_model": pipeline.small_model,
        "category_method": pipeline.category_method,
        "dataset": EXP["name"],
        "ollama": config.OLLAMA,
        "experiment": EXP,
        "vram_bytes_by_model": {**pipeline.large.loaded_models_vram(), **pipeline.small.loaded_models_vram()},
    }
    meta.update(extra or {})
    path = OUT / generation / "run_meta" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, default=str))


def _record(paper_id, route, fields, usage, wall, **extra):
    return {"arxiv_id": paper_id, "route": route, "fields": fields, "usage": usage,
            "extraction_wall_s": round(wall, 4), **extra}


def run_route(pipeline, route, ids, corpus, out_path, retrieve=None, label=""):
    done = _done_ids(out_path)
    todo = [i for i in ids if i not in done]
    print(f"[{label}] {len(done)} hechos, {len(todo)} pendientes -> {out_path}")
    t_start = time.perf_counter()
    for n, pid in enumerate(todo, 1):
        paper = corpus[pid]
        retrieved, cases, r_time = [], None, 0.0
        if retrieve is not None:
            t0 = time.perf_counter()
            hits = retrieve(pid)
            r_time = time.perf_counter() - t0
            cases = [c for _, c in hits]
            retrieved = [{"arxiv_id": c["arxiv_id"], "sim": round(s, 4)} for s, c in hits]
        try:
            fields, usage, wall = pipeline.run(route, paper, cases)
        except Exception as e:
            print(f"[{label}] error en {pid}: {e}")
            continue
        _append_jsonl(out_path, _record(pid, route, fields, usage, wall + r_time,
                                        retrieval_s=round(r_time, 5), retrieved=retrieved))
        if n % 25 == 0:
            rate = (time.perf_counter() - t_start) / n
            print(f"[{label}] {n}/{len(todo)}  ~{rate:.1f}s/paper, quedan ~{rate * (len(todo) - n) / 60:.0f} min")


# ---------------------------------------------------------------------------
# Fases
# ---------------------------------------------------------------------------
def phase_llm(args):
    corpus, split, _ = load_state()
    pipeline = make_pipeline(args)
    if not args.mock:
        warm_up(pipeline.large, pipeline.large_models.values())
    # Primero el test (línea base) y después solo la parte del pool que necesitan los umbrales.
    max_needed = max(EXP["thresholds"])
    pool_needed = set()
    for seed in EXP["seeds"]:
        order = list(split["pool"])
        random.Random(seed).shuffle(order)
        pool_needed.update(order[: int(max_needed * 1.05) + 5])  # margen por fallos
    ids = list(split["test"]) + sorted(pool_needed)
    run_route(pipeline, "llm", ids, corpus, pred_dir(args.generation) / "llm.jsonl", label="llm")
    save_run_meta(args.generation, "llm", pipeline)


def phase_small(args):
    corpus, split, _ = load_state()
    pipeline = make_pipeline(args)
    if not args.mock:
        warm_up(pipeline.small, [pipeline.small_model])
    run_route(pipeline, "small", split["test"], corpus, pred_dir(args.generation) / "small.jsonl", label="small")
    save_run_meta(args.generation, "small", pipeline)


def _case_from_prediction(rec, corpus):
    return {"arxiv_id": rec["arxiv_id"], "abstract": corpus[rec["arxiv_id"]]["abstract"],
            "fields": rec["fields"], "source": "llm"}


def _case_from_gt(paper, labelset):
    gt = paper["gt"]
    name = gt.get("name")
    if isinstance(name, list):
        name = name[0] if name else ""
    fields = {
        "name": name or "",
        "category": labelset.canonical(gt.get("category")) or "",
        "tasks": list(gt.get("tasks") or []),
        "tasks_raw": list(gt.get("tasks") or []),
        "repo": list(gt.get("repo") or []),
        "datasets": list(gt.get("datasets") or []),
        "metrics": list(gt.get("metrics") or []),
    }
    return {"arxiv_id": paper["arxiv_id"], "abstract": paper["abstract"], "fields": fields, "source": "gt"}


def _retriever(case_base, emb, k):
    def retrieve(pid):
        return case_base.retrieve(k=k, exclude_id=pid, query_embedding=emb[pid])
    return retrieve


def phase_cbr(args):
    corpus, split, emb = load_state()
    llm_preds = {r["arxiv_id"]: r for r in _read_jsonl(pred_dir(args.generation) / "llm.jsonl")}
    if not llm_preds:
        raise SystemExit("Faltan las predicciones LLM. Ejecuta primero la fase 'llm'.")
    pipeline = make_pipeline(args)
    if not args.mock:
        warm_up(pipeline.small, [pipeline.small_model])
    k = args.k or EXP["k"]

    for seed in EXP["seeds"]:
        order = list(split["pool"])
        random.Random(seed).shuffle(order)
        available = [pid for pid in order if pid in llm_preds]
        for thr in EXP["thresholds"]:
            if len(available) < thr:
                print(f"[cbr] AVISO: solo hay {len(available)} casos LLM en el pool para umbral {thr}.")
            chosen = available[:thr]
            cb = CaseBase(Embedder(EXP["embedder"]))
            cb.add_many([_case_from_prediction(llm_preds[p], corpus) for p in chosen],
                        embeddings=np.stack([emb[p] for p in chosen]))
            retrieve = _retriever(cb, emb, k)
            tag = f"t{thr}_s{seed}" + (f"_k{k}" if k != 1 else "")
            for route in ("retrieval_only", "cbr"):
                out = pred_dir(args.generation) / f"{route}_{tag}.jsonl"
                run_route(pipeline, route, split["test"], corpus, out, retrieve=retrieve,
                          label=f"{route} {tag}")
    save_run_meta(args.generation, "cbr", pipeline, {"k": k})


def phase_oracle(args):
    """Casos con GT, leave-one-out dentro del test: techo del método CBR."""
    corpus, split, emb = load_state()
    pipeline = make_pipeline(args)
    if not args.mock:
        warm_up(pipeline.small, [pipeline.small_model])
    k = args.k or EXP["k"]
    cb = CaseBase(Embedder(EXP["embedder"]))
    cb.add_many([_case_from_gt(corpus[p], pipeline.labelset) for p in split["test"]],
                embeddings=np.stack([emb[p] for p in split["test"]]))
    retrieve = _retriever(cb, emb, k)
    tag = "gtloo" + (f"_k{k}" if k != 1 else "")
    for route in ("retrieval_only", "cbr"):
        run_route(pipeline, route, split["test"], corpus,
                  pred_dir(args.generation) / f"{route}_{tag}.jsonl", retrieve=retrieve,
                  label=f"{route} {tag}")
    save_run_meta(args.generation, "oracle", pipeline, {"k": k})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["prepare", "llm", "small", "cbr", "oracle", "all"])
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=sorted(config.DATASETS),
                    help="perfil de dataset (config.DATASETS)")
    ap.add_argument("--generation", default="gen2026", choices=sorted(config.GENERATIONS))
    ap.add_argument("--pool-ocr-only", action="store_true",
                    help="usar como pool solo los papers con OCR hecho (OCR parcial por fases)")
    ap.add_argument("--thresholds", type=int, nargs="+", default=None, metavar="N",
                    help="tamaños de la base de casos (por defecto los de config.py); p. ej. --thresholds 100 500")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, metavar="S",
                    help="semillas de ordenación del pool (por defecto las de config.py); p. ej. --seeds 0")
    ap.add_argument("--k", type=int, default=None, help="casos recuperados (por defecto config.EXPERIMENT['k'])")
    ap.add_argument("--mock", action="store_true", help="backend simulado, sin GPU")
    ap.add_argument("--force", action="store_true", help="regenera corpus/partición/embeddings en 'prepare'")
    args = ap.parse_args()

    global EXP, OUT, CACHE, POOL_OCR_ONLY
    POOL_OCR_ONLY = args.pool_ocr_only
    EXP = config.dataset_profile(args.dataset)
    if args.thresholds:
        EXP["thresholds"] = sorted(set(args.thresholds))
    if args.seeds is not None:
        EXP["seeds"] = list(dict.fromkeys(args.seeds))
    base = EXP["out_dir"] + ("_mock" if args.mock else "")   # lo simulado nunca se mezcla con lo real
    OUT = Path(base) / EXP["name"]
    CACHE = OUT / "cache"
    print(f"[config] dataset={EXP['name']}  generación={args.generation}  resultados en {OUT}/")
    print(f"[config] umbrales={EXP['thresholds']}  semillas={EXP['seeds']}  k={args.k or EXP['k']}")
    if args.mock:
        print("[mock] backend simulado: resultados sin significado")

    if args.phase in ("prepare", "all"):
        prepare(args)
    if args.phase in ("llm", "all"):
        phase_llm(args)
    if args.phase in ("small", "all"):
        phase_small(args)
    if args.phase in ("cbr", "all"):
        phase_cbr(args)
    if args.phase in ("oracle", "all"):
        phase_oracle(args)


if __name__ == "__main__":
    main()
