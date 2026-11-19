import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.titlesize": 16
})

def analyze_cbr_transition(csv_path="resultados_detallados_cbr.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo '{csv_path}'.")
        return

    df = pd.read_csv(csv_path)
    
    # Aseguramos tipos
    df = df.dropna(subset=["threshold", "used_cbr", "paper_avg_f1"])
    df["threshold"] = df["threshold"].astype(int)
    df["used_cbr"] = df["used_cbr"].astype(bool)
    
    # Asignamos etiquetas legibles
    df["Regimen"] = df["used_cbr"].map({False: "Fase 1: Solo LLM (Warm-up)", True: "Fase 2: Con CBR"})

    # ---------------------------------------------------------
    # 1. TABLA COMPARATIVA CON DELTAS
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" ANÁLISIS DE IMPACTO: SOLO LLM vs. CBR ACTIVADO")
    print("="*70)
    
    grouped = df.groupby(["threshold", "used_cbr"]).agg({
        "paper_avg_f1": "mean",
        "paper_avg_precision": "mean",
        "paper_avg_recall": "mean",
        "generation_time_seconds": "mean",
        "arxiv_id": "count"
    }).unstack(level="used_cbr")

    for t in df["threshold"].unique():
        sub = df[df["threshold"] == t]
        llm_part = sub[sub["used_cbr"] == False]
        cbr_part = sub[sub["used_cbr"] == True]
        
        n_llm = len(llm_part)
        n_cbr = len(cbr_part)
        
        print(f"\n--- THRESHOLD = {t} ---")
        print(f"  * Papers evaluados solo con LLM : {n_llm}")
        print(f"  * Papers evaluados con CBR      : {n_cbr}")
        
        if n_cbr == 0:
            print("  [!] Este umbral aún no procesó suficientes papers para activar el CBR.")
            continue
            
        f1_llm = llm_part["paper_avg_f1"].mean()
        f1_cbr = cbr_part["paper_avg_f1"].mean()
        delta_f1 = f1_cbr - f1_llm
        
        time_llm = llm_part["generation_time_seconds"].mean()
        time_cbr = cbr_part["generation_time_seconds"].mean()
        delta_time = time_cbr - time_llm

        sign_f1 = "+" if delta_f1 >= 0 else ""
        sign_time = "+" if delta_time >= 0 else ""

        print(f"  * F1 Medio (Solo LLM) : {f1_llm:.4f}")
        print(f"  * F1 Medio (Con CBR)  : {f1_cbr:.4f}  -->  Diferencia: {sign_f1}{delta_f1:.4f} ({sign_f1}{(delta_f1/f1_llm)*100:.2f}%)")
        print(f"  * Tiempo Medio LLM    : {time_llm:.2f}s")
        print(f"  * Tiempo Medio CBR    : {time_cbr:.2f}s  -->  Diferencia: {sign_time}{delta_time:.2f}s ({sign_time}{(delta_time/time_llm)*100:.2f}%)")

    # ---------------------------------------------------------
    # 2. GRÁFICA COMPARATIVA ANTES VS DESPUÉS
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filtramos umbrales que tengan datos en ambas fases
    thresholds_with_both = []
    for t in df["threshold"].unique():
        if len(df[(df["threshold"] == t) & (df["used_cbr"] == True)]) > 0:
            thresholds_with_both.append(t)
            
    df_plot = df[df["threshold"].isin(thresholds_with_both)]

    if not df_plot.empty:
        # Gráfica 1: F1 Score
        sns.barplot(
            data=df_plot,
            x="threshold",
            y="paper_avg_f1",
            hue="Regimen",
            palette=["#4c72b0", "#55a868"],
            ci=None,
            ax=axes[0]
        )
        axes[0].set_title("Comparativa de Calidad (F1-Score)", fontweight="bold")
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("F1 Promedio")
        axes[0].set_ylim(0, 1.05)

        # Gráfica 2: Tiempo de Ejecución
        sns.barplot(
            data=df_plot,
            x="threshold",
            y="generation_time_seconds",
            hue="Regimen",
            palette=["#4c72b0", "#c44e52"],
            ci=None,
            ax=axes[1]
        )
        axes[1].set_title("Comparativa de Eficiencia (Segundos por Paper)", fontweight="bold")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Tiempo Medio (s)")

        plt.tight_layout()
        plt.savefig("comparativa_impacto_cbr.png", dpi=300)
        print("\nGráfica guardada como 'comparativa_impacto_cbr.png'")
        plt.show()
    else:
        print("\nNo hay suficientes datos con CBR activado para generar las gráficas comparativas.")

if __name__ == "__main__":
    analyze_cbr_transition()