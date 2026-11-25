"""
Ayuda para decidir el umbral de solapamiento con el que dos tareas abiertas cuentan como la misma.

Ningún umbral único sobre el solapamiento de palabras separa bien todos los pares (por ejemplo,
"node classification" / "semi-supervised node classification" y "knowledge graph completion" /
"knowledge graph embedding" puntúan igual). Por eso se decide con datos:

  1. Pares predicción <-> ground truth reales, agrupados por banda de solapamiento. Mira los de cada
     banda y decide a partir de qué banda ya son "la misma tarea".
  2. Cuánto cambia el F1 de tareas de cada ruta según el umbral. Si la ordenación de las rutas (llm, cbr,
     small...) no cambia con el umbral, la elección no afecta a la conclusión del experimento.

  python scripts/tasks_threshold_report.py                 # tras ejecutar el experimento
  python scripts/tasks_threshold_report.py --pairs 10      # más pares por banda
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from evaluate_predictions import load_scores  # noqa: E402
from utils.evaluation_tools import normalize_task, open_tasks, task_similarity  # noqa: E402

THRESHOLDS = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
BANDS = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5)]


def collect_pairs(out):
    """Para cada tarea predicha por la ruta llm, su tarea del GT más parecida (sin contar las idénticas)."""
    test = set(json.loads((out / "split.json").read_text())["test"])
    gt = {}
    with open(out / "cache" / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["arxiv_id"] in test:
                gt[p["arxiv_id"]] = p["gt"].get("tasks") or []
    pairs = []
    path = out / "gen2026" / "predictions" / "llm.jsonl"
    if not path.is_file():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["arxiv_id"] not in gt:
                continue
            for pred in open_tasks(r["fields"]):
                best = max(((task_similarity(pred, g), g) for g in gt[r["arxiv_id"]]), default=(0.0, None))
                if best[1] and normalize_task(pred) != normalize_task(best[1]):
                    pairs.append((best[0], pred, best[1]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=sorted(config.DATASETS))
    ap.add_argument("--results", default="results")
    ap.add_argument("--pairs", type=int, default=6, help="pares de ejemplo por banda")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    profile = config.dataset_profile(args.dataset)
    out = Path(args.results) / profile["name"]
    if not (out / "split.json").is_file():
        raise SystemExit(f"No hay resultados en {out}/. Ejecuta antes el experimento.")

    pairs = collect_pairs(out)
    print(f"\n=== 1. Pares predicción <-> GT más parecido (ruta llm, test; {len(pairs)} tareas predichas sin coincidencia exacta)")
    print("    Dos tareas con las mismas palabras ya cuentan como iguales; aquí solo van las que difieren.")
    rng = random.Random(args.seed)
    for lo, hi in BANDS:
        band = [p for p in pairs if lo <= p[0] < hi]
        print(f"\n--- solapamiento {lo:.1f} a {hi:.1f}: {len(band)} pares")
        for sim, pred, g in rng.sample(band, min(args.pairs, len(band))):
            print(f"    {sim:.2f}   predicha: {pred!r:48s} GT: {g!r}")
    print("\n    ¿A partir de qué banda dirías que ya es la misma tarea? Ese es el umbral.")

    print("\n=== 2. F1 de tareas por ruta según el umbral (media por paper de test)")
    rows = {}
    for th in THRESHOLDS:
        df, _ = load_scores(out, profile, task_threshold=th)
        for cfg, g in df.groupby("config", sort=False):
            rows.setdefault(cfg, {})[th] = g.groupby("arxiv_id")["tasks_f1"].mean().mean()
    print(f"    {'ruta':26s}" + "".join(f"{('>=' + str(t)):>8s}" for t in THRESHOLDS))
    for cfg, vals in rows.items():
        print(f"    {cfg:26s}" + "".join(f"{vals[t]:8.3f}" for t in THRESHOLDS))
    print("\n    Si las rutas se ordenan igual con cualquier umbral, elegirlo no cambia la conclusión.")


if __name__ == "__main__":
    main()
