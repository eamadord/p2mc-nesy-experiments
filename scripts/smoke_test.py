"""
Prueba de humo del pipeline con el modelo REAL de Ollama sobre unos pocos papers del test.

Es lo primero que hay que ejecutar antes de lanzar los experimentos: comprueba que Ollama responde con
JSON válido, que los campos no salen vacíos y cuánto tarda cada ruta. No sustituye al experimento.

  bash scripts/p2mc.sh smoke                 # 3 papers, rutas llm (modelos grandes) y small (modelo pequeño)
  bash scripts/p2mc.sh smoke --n 5 --generation gen2026
  python scripts/smoke_test.py --mock        # sin Ollama, para probar el propio script
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from model_card_generation_pipeline import FieldPipeline  # noqa: E402
from utils.evaluation_tools import score_paper  # noqa: E402

FIELDS = ["name", "category", "tasks", "repo", "datasets", "metrics"]


def short(v, n=58):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=sorted(config.DATASETS))
    ap.add_argument("--generation", default="gen2026", choices=sorted(config.GENERATIONS))
    ap.add_argument("--n", type=int, default=3, help="papers de test a probar")
    ap.add_argument("--mock", action="store_true", help="backend simulado, sin Ollama")
    args = ap.parse_args()

    profile = config.dataset_profile(args.dataset)
    out = Path(profile["out_dir"] + ("_mock" if args.mock else "")) / profile["name"]
    corpus_path, split_path = out / "cache" / "corpus.jsonl", out / "split.json"
    if not (corpus_path.is_file() and split_path.is_file()):
        raise SystemExit(f"Falta {corpus_path}. Ejecuta antes: bash scripts/p2mc.sh start prepare")
    test = set(json.loads(split_path.read_text())["test"])
    papers = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["arxiv_id"] in test:
                papers.append(p)
            if len(papers) >= args.n:
                break

    client = None
    if args.mock:
        from extractors.field_extractor import MockOllamaClient
        client = MockOllamaClient()
    pipe = FieldPipeline(generation=args.generation, client=client, profile=profile)

    if not args.mock:
        from experiment_cbr import warm_up
        print("Cargando los modelos en la GPU (no se cronometra)...", flush=True)
        warm_up(pipe.large, pipe.large_models.values())
        warm_up(pipe.small, [pipe.small_model])

    problems, summary = [], {}
    for route in ("llm", "small"):
        times, failed, empty_core, tokens = [], 0, 0, 0
        for paper in papers:
            fields, usage, wall = pipe.run(route, paper)
            scores = score_paper(fields, paper["gt"], labelset=pipe.labelset)
            times.append(wall); failed += usage["llm_failed_calls"]; tokens += usage["prompt_tokens"] + usage["gen_tokens"]
            print(f"\n{'=' * 78}\n[{route}] {paper['arxiv_id']}  {wall:.1f} s  llamadas={usage['llm_calls']} fallidas={usage['llm_failed_calls']}"
                  f"  tokens={usage['prompt_tokens']}+{usage['gen_tokens']}  tablas OCR: {'sí' if paper.get('tables_tsv') else 'NO'}")
            for fld in FIELDS:
                key = {"name": "name_relaxed", "category": "category_acc"}.get(fld, f"{fld}_f1")
                sc = scores.get(key)
                mark = "–" if sc is None or (isinstance(sc, float) and math.isnan(sc)) else ("✓" if sc >= 0.999 else ("~" if sc > 0 else "✗"))
                gt = paper["gt"].get(fld)
                print(f"  {fld:9s} {mark}  pred: {short(fields.get(fld))}\n{'':14s}GT:   {short(gt)}")
            empty_core += sum(1 for fld in ("name", "tasks", "datasets", "metrics") if not fields.get(fld))
        n = max(1, len(papers))
        summary[route] = {"s_per_paper": sum(times) / n, "failed": failed, "empty": empty_core / (n * 4), "tokens": tokens / n}
        if failed:
            problems.append(f"[{route}] {failed} llamadas fallidas (Ollama no devolvió un JSON válido)")
        if empty_core / (n * 4) > 0.5:
            problems.append(f"[{route}] más de la mitad de los campos principales salen vacíos")

    print(f"\n{'=' * 78}\nRESUMEN ({len(papers)} papers)")
    for route, v in summary.items():
        print(f"  {route:6s} {v['s_per_paper']:6.1f} s/paper | {v['tokens']:7.0f} tokens/paper | llamadas fallidas: {v['failed']} | campos principales vacíos: {100 * v['empty']:.0f}%")
    if "llm" in summary and "small" in summary and summary["small"]["s_per_paper"] > 0:
        r = summary["llm"]["s_per_paper"] / summary["small"]["s_per_paper"]
        print(f"  el modelo pequeño es {r:.1f}x más rápido que los grandes en estos papers")
    print("\nRESULTADO: " + ("OK: se puede lanzar el experimento." if not problems else "REVISAR ANTES DE LANZAR:\n  - " + "\n  - ".join(problems)))
    print("(mira también que 'pred' se parezca a 'GT': ✓ acierta, ~ acierta en parte, ✗ no acierta, – sin GT)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
