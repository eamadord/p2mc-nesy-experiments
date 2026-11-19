import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Cargar datos del CSV
df = pd.read_csv("resumen_metricas_cbr_bertscore.csv")

# 2. Configurar la figura con 2 paneles contiguos
fig, (ax_global, ax_fields) = plt.subplots(1, 2, figsize=(18, 6.5))

# -------------------------------------------------------------
# PANEL 1: Métricas Globales y Tiempo de Ejecución (Doble Eje Y)
# -------------------------------------------------------------
x_pos = np.arange(len(df["threshold"]))
threshold_labels = [f"Th = {t}" for t in df["threshold"]]

# Calidad semántica global en el eje Y izquierdo
line_f1 = ax_global.plot(
    x_pos,
    df["GLOBAL_F1"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#1f77b4",
    label="GLOBAL F1",
)
line_prec = ax_global.plot(
    x_pos,
    df["GLOBAL_Precision"],
    marker="s",
    linewidth=2,
    linestyle="--",
    markersize=7,
    color="#2ca02c",
    label="GLOBAL Precision",
)
line_rec = ax_global.plot(
    x_pos,
    df["GLOBAL_Recall"],
    marker="^",
    linewidth=2,
    linestyle=":",
    markersize=7,
    color="#9467bd",
    label="GLOBAL Recall",
)

ax_global.set_title(
    "A) Métricas Globales vs. Tiempo de Ejecución",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax_global.set_xlabel(
    "Umbral de activación CBR (casos)", fontsize=11, fontweight="bold"
)
ax_global.set_ylabel(
    "Puntuación BertScore (0 - 1)", fontsize=11, fontweight="bold"
)
ax_global.set_ylim(0.5, 1.0)
ax_global.set_xticks(x_pos)
ax_global.set_xticklabels(threshold_labels, fontsize=10)
ax_global.grid(True, linestyle="--", alpha=0.5)

# Tiempo medio por caso en el eje Y derecho
ax_time = ax_global.twinx()
bars_time = ax_time.bar(
    x_pos,
    df["avg_time_seconds"],
    width=0.35,
    alpha=0.25,
    color="#d62728",
    label="Tiempo medio (s)",
)
ax_time.set_ylabel(
    "Tiempo de respuesta promedio (s)",
    fontsize=11,
    fontweight="bold",
    color="#d62728",
)
ax_time.tick_params(axis="y", labelcolor="#d62728")
ax_time.set_ylim(0, df["avg_time_seconds"].max() * 1.3)

# Anotación del valor exacto de tiempo sobre cada barra
for idx, val in enumerate(df["avg_time_seconds"]):
    ax_time.annotate(
        f"{val:.2f} s",
        xy=(idx, val),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
    )

# Leyenda unificada para el panel global
all_plots = line_f1 + line_prec + line_rec + [bars_time]
all_labels = [p.get_label() for p in all_plots]
ax_global.legend(
    all_plots,
    all_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True,
)

# -------------------------------------------------------------
# PANEL 2: Rendimiento por Campo Individual y Escenario
# -------------------------------------------------------------
fields = [
    "model_f1",
    "cat_f1",
    "tasks_f1",
    "datasets_f1",
    "metrics_f1",
    "repo_f1",
]
clean_names = ["Model", "Category", "Tasks", "Datasets", "Metrics", "Repo"]

x_fields = np.arange(len(clean_names))
num_thresholds = len(df)
bar_width = 0.18
palette = ["#2b5c8f", "#4682b4", "#f39c12", "#e74c3c"]

for i, row in df.iterrows():
    offset = (i - (num_thresholds - 1) / 2) * bar_width
    scores = [row[f] for f in fields]
    ax_fields.bar(
        x_fields + offset,
        scores,
        width=bar_width,
        label=f"Th = {int(row['threshold'])} ({row['avg_time_seconds']:.1f} s)",
        color=palette[i % len(palette)],
    )

ax_fields.set_title(
    "B) Comparativa F1 por Campo y Umbral CBR",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax_fields.set_xlabel("Campos Evaluados", fontsize=11, fontweight="bold")
ax_fields.set_ylabel("F1 BertScore", fontsize=11, fontweight="bold")
ax_fields.set_xticks(x_fields)
ax_fields.set_xticklabels(clean_names, fontsize=10, fontweight="bold")
ax_fields.set_ylim(0, 1.05)
ax_fields.grid(axis="y", linestyle="--", alpha=0.6)
ax_fields.legend(
    title="Escenario (Umbral / Tiempo)",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True,
)

# Título global y ajuste de márgenes
plt.suptitle(
    "Evaluación Integral del Pipeline Dinámico: LLM Pura vs. CBR",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Guardar y mostrar
plt.savefig("analisis_completo_cbr_pipeline.png", dpi=300)
plt.show()