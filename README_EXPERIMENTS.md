# Experimentos P2MC-NeSy (CBR + LLM ligero)

**Hipótesis:** una vez acumulada una masa crítica de model cards (casos), un LLM pequeño guiado por el caso más similar mantiene la calidad de extracción del pipeline con LLMs grandes y reduce de forma significativa el tiempo y los recursos.

## Corpus

Hay dos perfiles de dataset (`config.DATASETS`), que se eligen con `--dataset`:

| Perfil | Corpus | Campo cerrado | Partición |
|---|---|---|---|
| `kge` | `testing_data/`: papers de Knowledge Graph Embedding (corpus original) | Categoría de modelo KGE (5 clases) | Test de 400: todos los papers con GT rico + muestra aleatoria |
| `hetero` | `testing_data_hetero/`: papers de varias áreas con **todos** los campos en el GT | Área de investigación de PwC | Fijada por `build_dataset.py`, estratificada por área |

### Construir el corpus heterogéneo

```bash
python build_dataset.py select      # GT desde pwc-archive + partición
python build_dataset.py pdfs        # PDFs de arXiv (3 s entre peticiones: ~1 h por cada 1.000 papers)
python build_dataset.py grobid      # TEI XML (GROBID en localhost:8070)
python build_dataset.py ocr         # tablas con LightOnOCR (GPU): ver "Tablas con LightOnOCR" más abajo
python build_dataset.py stats
```

**Orden obligatorio:** `experiment_cbr.py --dataset hetero` no arranca hasta que existan `testing_data_hetero/pwc_final.json` (fase `select`) y los XML de los papers (fases `pdfs` y `grobid`). Si falta algo, el programa lo dice y te indica qué ejecutar.

### Tablas con LightOnOCR

Los datasets y las métricas viven en las tablas de resultados, así que las entradas de `datasets` y `metrics` incluyen las tablas extraídas con LightOnOCR (`lightocr_jsons/`). Todos los papers deben tener el mismo tratamiento: si solo unos tuvieran tablas, las entradas no serían comparables entre ellos.

LightOnOCR procesa cada página como una imagen con un modelo de 1.000 millones de parámetros, así que es la fase más cara. Para que sea viable, por defecto solo se envían las páginas que **parecen tener una tabla**: las que en la capa de texto del PDF tienen un pie de tabla ("Table 3:") al inicio de línea y al menos 10 números, hasta 8 por paper (`--ocr-pages all` las procesa todas). Es una heurística comprobada con PDFs sintéticos, no con PDFs reales de arXiv: conviene revisar a mano unos cuantos papers.

```bash
/opt/conda/envs/p2mc-ocr/bin/python build_dataset.py ocr --ocr-limit 5   # primero 5 papers: imprime s/paper y la estimación en horas
/opt/conda/envs/p2mc-ocr/bin/python build_dataset.py ocr                 # luego todos; reanudable
```

**OCR por fases.** Si el OCR de todo el pool tarda demasiado, se puede hacer solo una parte y ampliarlo después. `--ocr-pool-limit N` procesa todo el test y los primeros N papers del pool **en orden equilibrado por área** (el pool de `pwc_final.json` está agrupado por área, y sus primeros N serían de una sola). El umbral mayor de la pasada necesita unos `1,05 × umbral + 5` papers de pool: 530 para los umbrales 100 y 500, y 2.105 para el 2.000.

```bash
# Fase 1: test + 530 del pool (la pasada piloto)
bash scripts/p2mc.sh start ocr --ocr-pool-limit 530
# ... cuando termine el OCR, y también GROBID y prune:
python experiment_cbr.py prepare --dataset hetero --force     # las tablas se guardan AHORA
bash scripts/p2mc.sh start experiment --pool-ocr-only         # el pool se limita a los papers con tablas
```

Tres cosas a tener en cuenta:

- `prepare` guarda en su caché las tablas que existan **en ese momento**: el OCR debe haber terminado antes, y si se amplía después hay que repetir `prepare --force`.
- `--pool-ocr-only` es necesario: sin él el experimento elige el pool al azar entre todos y la mayoría de los casos no tendrían tablas.
- Al ampliar el OCR (por ejemplo `--ocr-pool-limit 2105`), el pool cambia y con él los casos de cada umbral. Borra antes las predicciones de la ruta CBR (`results/hetero/<generación>/predictions/cbr_*` y `retrieval_only_*`); `llm.jsonl` y `small.jsonl` no dependen del pool y se reutilizan.

La fase es reanudable (se salta los JSON válidos y rehace los corruptos), sigue si un paper falla y no necesita GROBID, así que puede ejecutarse a la vez que él. El cargador del experimento lee tanto el formato de los JSON de `testing_data/` como el de `LightOcrParser` y el de esta fase.

Por defecto: las 8 áreas con más papers completos (sin "Methodology" ni "Miscellaneous"), 75 papers de test y 300 de pool por área. Se puede cambiar con `--areas`, `--n-test-per-area`, `--n-pool-per-area`, `--min-year` y `--repo-policy official`.

Un paper es **completo** si tiene ID de arXiv, abstract, nombre del modelo, al menos una tarea, un dataset, una métrica y un repositorio oficial o citado en el paper, y un área. El pool no exige completitud: solo alimenta la base de casos, que se construye con predicciones del LLM.

Respecto a `GAP-KGE/pwc_extraction.py`, el constructor corrige: el cruce de papers por URL exacta (ahora por ID de arXiv), las métricas que se reiniciaban en cada fila del leaderboard, los datasets que no se guardaban, las subtareas y subdatasets ignorados, y las rutas de PDF/XML que no se persistían. El nombre del modelo es el más frecuente en los leaderboards y el resto se guardan como alias válidos para la evaluación.

## Diseño

| Ruta | Qué hace | Para qué sirve |
|---|---|---|
| `llm` | Modelos grandes por campo (pipeline P2MC original) | Línea base |
| `cbr@N` | Modelo pequeño + caso más similar de una base de N casos generados por `llm` | Método propuesto |
| `small` | Modelo pequeño **sin** caso | Ablación: ¿aporta algo el caso? |
| `retrieval_only@N` | Copia los campos del caso más similar, sin LLM | Ablación: ¿aporta algo el LLM? |
| `cbr@oracle_gt` / `retrieval_only@oracle_gt` | Casos con ground truth (leave-one-out en test) | Techo del método |

Principios:

- **Test fijo**, evaluado con todas las rutas: comparaciones **pareadas** sobre los mismos papers.
- **Mismas entradas y prompts** por campo en todas las rutas (`utils/paper_loader.build_field_inputs`, `extractors/field_extractor.FIELD_SPECS`). Solo cambian el modelo y la presencia del caso.
- **Mismo post-procesado simbólico** en todas las rutas: alineación de tareas con LPWC (`TaskMatcher`) y taxonomía cerrada de categorías.
- La base de casos de umbral N se construye con las N primeras predicciones `llm` de una ordenación aleatoria del pool; se repite con 3 semillas.
- Los tiempos excluyen la carga de modelos (calentamiento previo, `keep_alive`) y las consultas de red (SemOpenAlex, LPWC), que no forman parte de la extracción.

## Entorno

Todas las dependencias están fijadas en `environment.yml` (solo conda-forge, Python 3.11). Ollama no va dentro: se instala y se lanza aparte.

```bash
conda env create -f environment.yml
conda activate p2mc-nesy
```

- `scikit-learn` está fijado a **1.5.0 exacto** porque `utils/taxonomy_classifier.joblib` se serializó con esa versión; con cualquier otra scikit-learn avisa al cargarlo.
- Se instala `pytorch-cpu`: basta para los embeddings de la base de casos (all-MiniLM-L6-v2). Solo la fase `build_dataset.py ocr` (LightOnOCR) necesita GPU, y usa un entorno aparte, `environment-ocr.yml` (unos 4 GB), para no tocar el de los experimentos. No lo crees clonando `p2mc-nesy` e instalando `pytorch-gpu`: el solver clásico de conda se queda atascado en "solving environment". Con micromamba resuelve en segundos:

  ```bash
  conda create -n mm -c conda-forge micromamba -y
  # En redes que cortan muchas conexiones pequeñas ("SSL connect error ... Connection reset by peer"):
  export MAMBA_USE_SHARDED_REPODATA=false MAMBA_DOWNLOAD_THREADS=1 MAMBA_REMOTE_CONNECT_TIMEOUT_SECS=60
  /opt/conda/envs/mm/bin/micromamba create -r /opt/conda -n p2mc-ocr -f environment-ocr.yml -y
  /opt/conda/envs/p2mc-ocr/bin/python -c "import torch; print(torch.cuda.is_available())"   # debe dar True
  ```

  `MAMBA_USE_SHARDED_REPODATA` debe valer `false` (con `0` micromamba da un error de conversión): así descarga el índice completo del canal en vez de cientos de ficheros pequeños, que es lo que cortaba la red.

  El `/opt/conda` de esas órdenes es la instalación de conda de este servidor (`conda info --base`). Después, la fase se lanza con `/opt/conda/envs/p2mc-ocr/bin/python build_dataset.py ocr`.
- Actualizar el entorno tras cambiar `environment.yml`: `conda env update -f environment.yml --prune`.
- Para congelar las versiones exactas con las que salieron los resultados del paper, súbelas al repo:
  ```bash
  conda list --explicit > conda-explicit-linux-64.txt     # recrea el entorno idéntico: conda create -n p2mc-nesy --file conda-explicit-linux-64.txt
  pip freeze | grep -i ollama                            # versión exacta del cliente de Ollama
  ```
- Si `conda` tarda mucho en resolver, `mamba`/`micromamba` aceptan los mismos comandos.

## Sesiones que se cierran

En el servidor, cada cierto tiempo la sesión se cierra y se pierde todo lo que no está en el disco personal: los entornos de `/opt/conda` y los procesos en marcha. `scripts/p2mc.sh` lo resuelve:

- Los entornos y la caché de paquetes viven en `~/micromamba` (persistente). Recrear un entorno tras un cierre enlaza desde la caché, sin volver a descargar.
- Cada fase corre bajo un supervisor que la reintenta si falla y que la recuerda como "pendiente" hasta que termina bien.

```bash
cd ~/p2mc-nesy-experiments && git pull
bash scripts/p2mc.sh bootstrap                # una vez: crea p2mc-nesy, p2mc-ocr y grobid (en el disco persistente)

bash scripts/p2mc.sh start pdfs
bash scripts/p2mc.sh start grobid             # arranca antes el servidor GROBID y espera a que escuche
bash scripts/p2mc.sh start ocr --ocr-test-only --ocr-max-pages 3
bash scripts/p2mc.sh start experiment         # arranca antes Ollama y espera a que escuche
```

**Tras cada cierre de sesión**, un solo comando recrea los entornos que falten y relanza lo que no había terminado:

```bash
cd ~/p2mc-nesy-experiments && bash scripts/p2mc.sh resume
```

Otras órdenes: `status` (qué corre, qué datos hay, qué falta), `logs <fase>`, `stop [fase|all]`. Fases: `ollama`, `grobid-server`, `pdfs`, `grobid`, `ocr`, `prune`, `experiment`. Los logs quedan en `logs/`.

Todo lo que debe sobrevivir tiene que estar en tu carpeta personal: el repositorio, `~/micromamba`, `~/grobid` (compilado), `~/.local` (Ollama), `~/.ollama` (modelos) y `~/.cache/huggingface`. Comprueba tras el primer cierre que `ls ~/micromamba` sigue ahí; si no, ese disco no es persistente y hay que apuntar `P2MC_MAMBA_ROOT` a otro que lo sea.

## Lanzar los experimentos (tras el OCR)

Todo con `bash scripts/p2mc.sh` desde la raíz del repo. Cada paso tiene su comprobación: no pases al siguiente si no se cumple.

```bash
# 0. Estado y GPU libre. Las mediciones de tiempo solo valen si nada más usa la GPU.
bash scripts/p2mc.sh status
bash scripts/p2mc.sh stop ocr

# 1. GROBID y limpieza del corpus (si aún no está hecho: PDFs y XML deben ser casi los mismos)
bash scripts/p2mc.sh start grobid               # reanudable
bash scripts/p2mc.sh start prune --check-titles # descarta papers cuyo PDF no es el del ground truth
bash scripts/p2mc.sh stop grobid-server         # ya no hace falta: libera CPU

# 2. Preparar: guarda las tablas del OCR y calcula los embeddings
bash scripts/p2mc.sh start prepare              # debe mostrar "con OCR: test=N/N pool=M/M"

# 3. Prueba de humo con el Ollama real (3 papers, rutas llm y small)
bash scripts/p2mc.sh smoke                      # debe acabar en "RESULTADO: OK"

# 4. Pasada piloto: 2 umbrales y 1 semilla
bash scripts/p2mc.sh start experiment --thresholds 100 500 --seeds 0
bash scripts/p2mc.sh logs experiment

# 5. Evaluar (vale con resultados parciales)
bash scripts/p2mc.sh evaluate
cat results/hetero/summary.md
```

La pasada piloto procesa con la ruta `llm` unos 1.100 papers (el test y unos 530 del pool). La prueba de humo da los segundos por paper de cada ruta: multiplícalos para estimar el tiempo. Para ampliar a la configuración completa, relanza sin `--thresholds` ni `--seeds` (usa los de `config.py`) o con `--thresholds 100 500 1000 2000 --seeds 0 1 2`. Lo ya calculado se reutiliza, pero hay que borrar antes las predicciones `cbr_*` y `retrieval_only_*` si cambia el pool. Con el OCR de todos los papers hecho no hace falta `--pool-ocr-only`; `prepare` muestra cuántos tienen OCR.

## Cómo ejecutarlo

```bash
conda env create -f environment.yml && conda activate p2mc-nesy
ollama pull qwen3.5:9b && ollama pull qwen3.5:2b          # gen2026
ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull qwen2.5:1.5b   # gen2024

python experiment_cbr.py prepare --dataset hetero        # corpus, partición y embeddings (una vez)
python experiment_cbr.py all --dataset hetero --generation gen2026
python experiment_cbr.py all --dataset hetero --generation gen2024   # opcional: modelos de 2024
python evaluate_predictions.py --dataset hetero
# Ídem con --dataset kge para el corpus original
```

Todas las fases guardan cada predicción en cuanto se genera y se pueden **interrumpir y reanudar**. Para probar el flujo completo sin GPU: `python experiment_cbr.py all --dataset kge --mock` y `python evaluate_predictions.py --dataset kge --results results_mock` (resultados sin significado).

Recomendado para que los tiempos sean comparables: lanzar Ollama con `OLLAMA_NUM_PARALLEL=1` y sin otros procesos usando la GPU.

**Coste orientativo:** la fase `llm` recorre ~2.300 papers × 5 llamadas; la fase `cbr` recorre 400 papers × 4 umbrales × 3 semillas con el modelo pequeño. El script imprime la velocidad real y el tiempo restante cada 25 papers: úsalo para planificar.

## Resultados (`results/<perfil>/`)

| Fichero | Contenido |
|---|---|
| `<gen>/predictions/*.jsonl` | Predicción cruda por paper, uso de tokens y tiempos. Permite recalcular cualquier métrica sin volver a inferir |
| `<gen>/run_meta/*.json` | Modelos, parámetros y VRAM por modelo (`ollama ps`) de cada ejecución |
| `scores_per_paper.csv` | Puntuación por paper, configuración y semilla |
| `summary.csv`, `summary.md` | Media por campo con nº de papers evaluables, IC95% bootstrap y desviación entre semillas |
| `paired_vs_llm.csv` | Diferencia pareada con `llm`, IC95%, Wilcoxon y test de **no inferioridad** (margen `--margin`, 0.05 por defecto) |
| `resources.csv` | Tiempo de extracción, cómputo LLM, tokens y llamadas por paper, también en proporción respecto a `llm` |
| `f1_vs_threshold_<gen>.png` | Puntuación por campo frente al tamaño de la base de casos |
| `summary_by_area.csv` | Solo `hetero`: puntuación por área y fracción de casos recuperados que son del mismo área |

## Métricas

Exact match sobre entidades normalizadas, por campo y con su n (no se promedian campos con distinta cobertura):

- **name**: `name_relaxed` (una forma normalizada contiene a la otra); también `name_exact` y `name_token_f1`. `--bertscore` añade BERTScore reescalado.
- **category**: accuracy sobre el vocabulario cerrado del perfil (5 categorías KGE o las áreas elegidas).
- **tasks**: F1 contra todas las tareas del GT y `tasks_invocab_f1` contra las del GT que están en el vocabulario de tareas del perfil. En `kge` ese vocabulario tiene 20 tareas y cubre el 52% de las menciones del GT; en `hetero` es el vocabulario completo de PwC.
- **repo**: F1 sobre URLs normalizadas (`github.com/owner/repo`).
- **datasets / metrics**: F1 sobre nombres normalizados con alias (MR = Mean Rank, Hits@10 = H@10 = HR@10...).

Un campo sin GT no se evalúa (NaN); un campo con GT y predicción vacía puntúa 0.

## Limitaciones conocidas que conviene declarar

- `kge`: solo 158 papers tienen GT de nombre, datasets y métricas, y 62 de categoría.
- `hetero`: exigir todos los campos sesga hacia papers bien documentados y orientados a benchmarks. Los leaderboards de PwC no recogen todos los datasets que usa cada paper, lo que penaliza la precisión. Conviene anotar a mano una muestra para estimar la calidad del GT.
- Contaminación: los modelos de 2026 pueden haber visto estos papers y el propio PwC. Reportad los resultados por año (`date` en `pwc_final.json`).
- El vocabulario de tareas (20 tareas LPWC) limita el recall de tareas; ampliarlo con la lista completa de LPWC mejoraría ambas rutas.
- GROBID a veces parte las URLs con espacios (`github.com/dalab/hyperbolic_ cones`), lo que penaliza el campo repo en todas las rutas.
- El clasificador de categorías (`utils/taxonomy_classifier.joblib`) se entrenó con `dataset_con_rutas_xml.json`: hay que documentar con qué papers, por si se solapan con los 62 del test. Se serializó con scikit-learn 1.5.
