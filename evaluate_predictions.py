"""
Evaluación de las predicciones de experiment_cbr.py.

Salidas (en results/<generación>/ y results/):
  scores_per_paper.csv   una fila por (configuración, semilla, paper)
  summary.csv            media por campo, nº de papers evaluables, IC95% bootstrap, std entre semillas
  paired_vs_llm.csv      diferencia pareada con la ruta LLM (mismos papers), IC95%,
                         Wilcoxon y test de NO INFERIORIDAD con margen configurable
  resources.csv          tiempo de extracción, cómputo LLM, tokens y llamadas por paper
  summary.md             tablas listas para revisar
  f1_vs_threshold_<gen>.png
  summary_by_area.csv    (solo si el corpus tiene áreas) puntuación por área de investigación
                         y fracción de casos recuperados que son del mismo área

Uso:
  python evaluate_predictions.py --dataset hetero
  python evaluate_predictions.py --dataset kge --results results_mock
  python evaluate_predictions.py --margin 0.03 --bertscore
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils.evaluation_tools import DEFAULT_TASK_THRESHOLD, PRIMARY_METRIC, bertscore_name, score_paper

RESOURCE_COLS = ["extraction_wall_s", "llm_compute_s", "llm_load_s", "prompt_tokens",
                 "gen_tokens", "llm_calls", "llm_failed_calls", "retrieval_s"]
FIELD_ORDER = ["name", "category", "tasks", "tasks_exact", "repo", "datasets", "metrics"]

_CFG_RE = re.compile(r"^(?P<route>retrieval_only|cbr)_(?:(?P<gtloo>gtloo)|t(?P<thr>\d+)_s(?P<seed>\d+))(?:_k(?P<k>\d+))?$")


def parse_config(stem):
    if stem in ("llm", "small"):
        return {"config": stem, "route": stem, "threshold": None, "seed": 0, "k": None}
    m = _CFG_RE.match(stem)
    if not m:
        return None
    route = m.group("route")
    k = int(m.group("k")) if m.group("k") else 1
    if m.group("gtloo"):
        cfg = f"{route}@oracle_gt" + (f"_k{k}" if k != 1 else "")
        return {"config": cfg, "route": route, "threshold": "oracle", "seed": 0, "k": k}
    thr = int(m.group("thr"))
    cfg = f"{route}@{thr}" + (f"_k{k}" if k != 1 else "")
    return {"config": cfg, "route": route, "threshold": thr, "seed": int(m.group("seed")), "k": k}


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(diffs):
    d = np.asarray(diffs, dtype=float)
    if len(d) < 5 or np.allclose(d, 0):
        return math.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d, zero_method="zsplit").pvalue)
    except Exception:
        return math.nan


def load_scores(results_dir, profile, use_bertscore=False, task_threshold=DEFAULT_TASK_THRESHOLD):
    results_dir = Path(results_dir)
    corpus_path = results_dir / "cache" / "corpus.jsonl"
    split = json.loads((results_dir / "split.json").read_text())
    test = set(split["test"])
    gt, area = {}, {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            area[p["arxiv_id"]] = p.get("area")
            if p["arxiv_id"] in test:
                gt[p["arxiv_id"]] = p["gt"]
    labelset = config.category_labelset(profile)

    rows = []
    for gen_dir in sorted(d for d in results_dir.iterdir() if (d / "predictions").is_dir()):
        for path in sorted((gen_dir / "predictions").glob("*.jsonl")):
            cfg = parse_config(path.stem)
            if cfg is None:
                print(f"[eval] ignorando {path.name}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    pid = r["arxiv_id"]
                    if pid not in gt:      # llm.jsonl también contiene el pool
                        continue
                    s = score_paper(r["fields"], gt[pid], labelset=labelset, task_threshold=task_threshold)
                    if use_bertscore:
                        s["name_bertscore"] = bertscore_name(r["fields"].get("name"), gt[pid].get("name"))
                    u = r.get("usage", {})
                    same_area = math.nan
                    if r.get("retrieved") and area.get(pid):
                        hits = [area.get(c["arxiv_id"]) == area[pid] for c in r["retrieved"]]
                        same_area = sum(hits) / len(hits)
                    rows.append({
                        "generation": gen_dir.name, **cfg, "arxiv_id": pid, "area": area.get(pid),
                        "retrieved_same_area": same_area, **s,
                        "extraction_wall_s": r.get("extraction_wall_s"),
                        "retrieval_s": r.get("retrieval_s", 0.0),
                        **{c: u.get(c) for c in RESOURCE_COLS if c in u},
                    })
    df = pd.DataFrame(rows)
    return df, len(test)


def _metric_cols(df):
    return [c for c in df.columns if re.match(r"^(name|category|tasks|repo|datasets|metrics)", c)]


def summarize(df, n_test):
    metric_cols = _metric_cols(df)
    out = []
    for (gen, cfg), g in df.groupby(["generation", "config"], sort=False):
        per_paper = g.groupby("arxiv_id")[metric_cols].mean()          # media entre semillas
        per_seed = g.groupby("seed")[metric_cols].mean()                # media por semilla
        row = {"generation": gen, "config": cfg, "route": g["route"].iloc[0],
               "threshold": g["threshold"].iloc[0], "n_seeds": g["seed"].nunique(),
               "papers_predicted": per_paper.shape[0], "papers_in_test": n_test}
        for c in metric_cols:
            vals = per_paper[c].dropna()
            row[c] = vals.mean() if len(vals) else math.nan
            row[f"{c}__n"] = len(vals)
            if c in PRIMARY_METRIC.values():
                lo, hi = bootstrap_ci(vals.values)
                row[f"{c}__ci_lo"], row[f"{c}__ci_hi"] = lo, hi
                row[f"{c}__seed_std"] = per_seed[c].std() if len(per_seed) > 1 else math.nan
        out.append(row)
    return pd.DataFrame(out)


def paired_vs_llm(df, margin):
    rows = []
    for gen, g in df.groupby("generation"):
        base = g[g["config"] == "llm"].groupby("arxiv_id")[list(PRIMARY_METRIC.values())].mean()
        if base.empty:
            continue
        for cfg, h in g[g["config"] != "llm"].groupby("config", sort=False):
            other = h.groupby("arxiv_id")[list(PRIMARY_METRIC.values())].mean()
            for field, col in PRIMARY_METRIC.items():
                j = pd.concat([base[col].rename("llm"), other[col].rename("other")], axis=1).dropna()
                if j.empty:
                    continue
                d = (j["other"] - j["llm"]).values
                lo, hi = bootstrap_ci(d)
                rows.append({
                    "generation": gen, "config": cfg, "field": field, "metric": col, "n_paired": len(d),
                    "llm_mean": j["llm"].mean(), "config_mean": j["other"].mean(),
                    "diff": d.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "wilcoxon_p": wilcoxon_p(d),
                    f"non_inferior_margin_{margin}": bool(lo > -margin) if not math.isnan(lo) else None,
                })
    return pd.DataFrame(rows)


def by_area(df):
    """Métrica principal por área y fracción de casos recuperados del mismo área."""
    if "area" not in df or df["area"].isna().all():
        return pd.DataFrame()
    cols = list(PRIMARY_METRIC.values()) + ["retrieved_same_area"]
    per_paper = df.groupby(["generation", "config", "area", "arxiv_id"])[cols].mean().reset_index()
    out = per_paper.groupby(["generation", "config", "area"])[cols].mean()
    out["n_papers"] = per_paper.groupby(["generation", "config", "area"])["arxiv_id"].nunique()
    return out.reset_index()


def resources(df):
    cols = [c for c in RESOURCE_COLS if c in df.columns]
    res = df.groupby(["generation", "config"], sort=False)[cols].mean().reset_index()
    for gen in res["generation"].unique():
        base = res[(res["generation"] == gen) & (res["config"] == "llm")]
        if base.empty:
            continue
        for c in ("extraction_wall_s", "llm_compute_s", "prompt_tokens"):
            if c in res and base[c].iloc[0]:
                m = res["generation"] == gen
                res.loc[m, f"{c}__vs_llm"] = res.loc[m, c] / base[c].iloc[0]
    return res


def plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for gen, g in summary.groupby("generation"):
        fields = [f for f in FIELD_ORDER if PRIMARY_METRIC[f] in g.columns]
        fig, axes = plt.subplots(1, len(fields), figsize=(3.2 * len(fields), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        numeric = g[g["threshold"].apply(lambda t: isinstance(t, (int, np.integer)) or str(t).isdigit())]
        for ax, f in zip(axes, fields):
            col = PRIMARY_METRIC[f]
            for route, style in (("cbr", "-o"), ("retrieval_only", "--s")):
                s = numeric[numeric["route"] == route].copy()
                if s.empty:
                    continue
                s["thr"] = s["threshold"].astype(int)
                s = s.sort_values("thr")
                ax.plot(s["thr"], s[col], style, label=route)
                if f"{col}__ci_lo" in s:
                    ax.fill_between(s["thr"], s[f"{col}__ci_lo"], s[f"{col}__ci_hi"], alpha=0.15)
            for cfg, ls in (("llm", "k"), ("small", "gray")):
                b = g[g["config"] == cfg]
                if not b.empty:
                    ax.axhline(b[col].iloc[0], color=ls, linestyle=":", label=cfg)
            ax.set_title(f)
            ax.set_xlabel("casos en la base")
            ax.set_xscale("log")
        axes[0].set_ylabel("puntuación")
        axes[-1].legend(fontsize=8)
        fig.suptitle(gen)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"f1_vs_threshold_{gen}.png", dpi=150)
        plt.close(fig)


def write_markdown(summary, paired, res, path, margin):
    lines = ["# Resultados P2MC-NeSy", ""]
    prim = [PRIMARY_METRIC[f] for f in FIELD_ORDER if PRIMARY_METRIC[f] in summary.columns]
    for gen, g in summary.groupby("generation"):
        lines += [f"## {gen}", "", "### Calidad (media por paper; n = papers evaluables)", ""]
        header = "| config | " + " | ".join(prim) + " |"
        lines += [header, "|" + "---|" * (len(prim) + 1)]
        for _, r in g.iterrows():
            cells = []
            for c in prim:
                v, n = r.get(c), r.get(f"{c}__n")
                cells.append("—" if pd.isna(v) else f"{v:.3f} (n={int(n)})")
            lines.append(f"| {r['config']} | " + " | ".join(cells) + " |")
        rg = res[res["generation"] == gen]
        if not rg.empty:
            lines += ["", "### Recursos por paper", "", rg.to_markdown(index=False, floatfmt=".3f")]
        pg = paired[paired["generation"] == gen] if not paired.empty else paired
        if not pg.empty:
            lines += ["", f"### Diferencia pareada con la ruta LLM (no inferioridad, margen {margin})", "",
                      pg.drop(columns=["generation"]).to_markdown(index=False, floatfmt=".3f")]
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=sorted(config.DATASETS))
    ap.add_argument("--results", default="results", help="carpeta base (results o results_mock)")
    ap.add_argument("--margin", type=float, default=0.05, help="margen de no inferioridad (puntos absolutos)")
    ap.add_argument("--task-threshold", type=float, default=DEFAULT_TASK_THRESHOLD,
                    help="solapamiento mínimo (Dice, 0-1) para dar dos tareas por la misma; 1 = mismas palabras. "
                         "Provisional: decídelo con scripts/tasks_threshold_report.py")
    ap.add_argument("--bertscore", action="store_true", help="añade BERTScore reescalado para el nombre")
    args = ap.parse_args()

    profile = config.dataset_profile(args.dataset)
    out = Path(args.results) / profile["name"]
    if not (out / "split.json").is_file():
        raise SystemExit(
            f"No hay resultados en {out}/ (falta split.json).\n"
            f"Ejecuta primero: python experiment_cbr.py all --dataset {profile['name']} "
            "[--generation gen2026]")
    df, n_test = load_scores(out, profile, use_bertscore=args.bertscore, task_threshold=args.task_threshold)
    print(f"[eval] tareas: coincidencia por solapamiento >= {args.task_threshold} (columna tasks_f1) y exacta tras normalizar (tasks_exact_f1)")
    if df.empty:
        raise SystemExit("No hay predicciones que evaluar.")
    df.to_csv(out / "scores_per_paper.csv", index=False)
    summary = summarize(df, n_test)
    summary.to_csv(out / "summary.csv", index=False)
    paired = paired_vs_llm(df, args.margin)
    paired.to_csv(out / "paired_vs_llm.csv", index=False)
    res = resources(df)
    res.to_csv(out / "resources.csv", index=False)
    areas = by_area(df)
    if not areas.empty:
        areas.to_csv(out / "summary_by_area.csv", index=False)
    try:
        write_markdown(summary, paired, res, out / "summary.md", args.margin)
    except ImportError:
        print("[eval] instala 'tabulate' para generar summary.md")
    plot(summary, out)

    cols = ["generation", "config"] + [PRIMARY_METRIC[f] for f in FIELD_ORDER if PRIMARY_METRIC[f] in summary]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(3).to_string(index=False))
        print()
        print(res.round(3).to_string(index=False))
    print(f"\nResultados escritos en {out}/")


if __name__ == "__main__":
    main()
