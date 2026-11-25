#!/usr/bin/env bash
# scripts/p2mc.sh: entornos persistentes y relanzado de las fases largas.
#
# Problema que resuelve: en el servidor la sesión se cierra cada cierto tiempo y se pierde todo lo que
# no está en el disco personal ($HOME): los entornos de /opt/conda y los procesos en marcha.
#
# Solución:
#   * Entornos y caché de paquetes en $HOME/micromamba (persistente). Recrear un entorno tras un cierre
#     no vuelve a descargar nada: enlaza desde la caché.
#   * Un "supervisor" por fase que reintenta si el proceso falla.
#   * Una lista de fases "pendientes de terminar": tras cada cierre de sesión basta con
#         bash scripts/p2mc.sh resume
#
# Uso:
#   bash scripts/p2mc.sh bootstrap [entorno...]   crea los entornos que falten (p2mc-nesy p2mc-ocr grobid)
#   bash scripts/p2mc.sh start <fase> [args...]   lanza una fase en segundo plano con reintentos
#   bash scripts/p2mc.sh resume                   relanza las fases que no habían terminado
#   bash scripts/p2mc.sh status                   qué está en marcha, qué datos hay y qué falta
#   bash scripts/p2mc.sh smoke [--n 3]            prueba de humo del pipeline con el Ollama real
#   bash scripts/p2mc.sh evaluate                 evalúa las predicciones guardadas (vale con resultados parciales)
#   bash scripts/p2mc.sh logs <fase>              sigue el log de una fase
#   bash scripts/p2mc.sh stop [fase|all]          para una fase (y la quita de pendientes)
#
# Fases: ollama grobid-server pdfs grobid ocr prune prepare experiment
# Ejemplos:
#   bash scripts/p2mc.sh start ocr --ocr-test-only --ocr-max-pages 3
#   bash scripts/p2mc.sh start grobid           # arranca antes GROBID y espera a que escuche
#   bash scripts/p2mc.sh start experiment       # arranca antes Ollama y espera a que escuche
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/scripts/p2mc.sh"
ROOT="${P2MC_MAMBA_ROOT:-$HOME/micromamba}"
BINDIR="${P2MC_BIN:-$HOME/bin}"
MM="$BINDIR/micromamba"
MM_VERSION="${P2MC_MM_VERSION:-2.10.0-0}"
STATE="${P2MC_STATE:-$HOME/.p2mc}"
LOGS="${P2MC_LOGS:-$REPO/logs}"
GROBID_DIR="${GROBID_DIR:-$HOME/grobid}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/bin/ollama}"
DATASET="${P2MC_DATASET:-hetero}"
GENERATION="${P2MC_GENERATION:-gen2026}"
DATA_DIR="${P2MC_DATA_DIR:-$REPO/testing_data_hetero}"
PHASES="ollama grobid-server pdfs grobid ocr prune prepare experiment"   # orden de reanudación
ENVS="p2mc-nesy p2mc-ocr grobid"

# Variables que deben sobrevivir a los cierres de sesión y aplicarse a todas las fases (por ejemplo
# LIGHTONOCR_ATTN=sdpa). Un "export" en el terminal se pierde al cerrarse la sesión y "resume" no
# lo recordaría; las que están en este fichero, sí:
#     echo 'export LIGHTONOCR_ATTN=sdpa' >> ~/.p2mc/env
# shellcheck disable=SC1091
[ -f "$STATE/env" ] && . "$STATE/env"

# Los paquetes instalados con "pip --user" (~/.local) tapan a los del entorno y rompen la
# reproducibilidad: con esto, cada fase usa solo lo que fija su environment*.yml.
export PYTHONNOUSERSITE=1

# Redes que cortan muchas conexiones pequeñas: índice completo del canal y una descarga a la vez.
export MAMBA_ROOT_PREFIX="$ROOT"
export MAMBA_USE_SHARDED_REPODATA=false MAMBA_DOWNLOAD_THREADS=1 MAMBA_REMOTE_CONNECT_TIMEOUT_SECS=60

say() { printf '[p2mc %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------------------------ entornos
env_dir() { echo "$ROOT/envs/$1"; }
env_ok() {
  case "$1" in
    grobid) [ -x "$(env_dir grobid)/lib/jvm/bin/java" ] ;;
    *)      [ -x "$(env_dir "$1")/bin/python" ] ;;
  esac
}

get_mm() {
  [ -x "$MM" ] && return 0
  say "descargando micromamba $MM_VERSION a $BINDIR ..."
  mkdir -p "$BINDIR"
  local py; py="$(command -v python3 || command -v python || true)"
  [ -n "$py" ] || { say "no hay python3 para descargar micromamba"; return 1; }
  "$py" - "$MM_VERSION" "$BINDIR" <<'PY' || { say "no se pudo descargar micromamba"; return 1; }
import io, os, sys, tarfile, urllib.request
version, dest = sys.argv[1:3]
url = f"https://conda.anaconda.org/conda-forge/linux-64/micromamba-{version}.tar.bz2"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
    member = tar.getmember("bin/micromamba")
    member.name = "micromamba"
    tar.extract(member, dest)
os.chmod(os.path.join(dest, "micromamba"), 0o755)
PY
  [ -x "$MM" ]
}

ensure_env() {
  local name="$1" i
  env_ok "$name" && return 0
  get_mm || return 1
  say "creando el entorno '$name' en $ROOT/envs (si ya se descargó antes, sale de la caché)"
  for i in 1 2 3 4 5; do
    rm -rf "$(env_dir "$name")"
    case "$name" in
      p2mc-nesy) "$MM" create -n "$name" -f "$REPO/environment.yml" -y ;;
      p2mc-ocr)  "$MM" create -n "$name" -f "$REPO/environment-ocr.yml" -y ;;
      grobid)    "$MM" create -n "$name" -c conda-forge --override-channels openjdk=11 -y ;;
      *)         say "entorno desconocido: $name"; return 1 ;;
    esac
    if env_ok "$name"; then say "entorno '$name' listo"; return 0; fi
    say "intento $i fallido; reintento en 20 s"; sleep 20
  done
  say "no se pudo crear el entorno '$name'"; return 1
}

# ----------------------------------------------------------------------------------- servicios
port_open() { (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

wait_port() {
  local port="$1" secs="$2" t=0
  while ! port_open "$port"; do
    [ "$t" -ge "$secs" ] && return 1
    sleep 5; t=$((t + 5))
  done
}

need_service() {      # need_service <fase> <puerto> <segundos de espera>
  [ "${P2MC_SKIP_SERVICES:-0}" = 1 ] && return 0
  port_open "$2" && return 0
  say "el servicio '$1' no escucha en el puerto $2: lo arranco"
  cmd_start "$1"
  wait_port "$2" "$3" || { say "'$1' no ha abierto el puerto $2 en $3 s (mira $LOGS/$1.log)"; return 1; }
}

# ------------------------------------------------------------------------------------- fases
pidfile() { echo "$STATE/pids/$1.pid"; }

running() {
  local pf pid; pf="$(pidfile "$1")"
  [ -f "$pf" ] || return 1
  pid="$(cat "$pf" 2>/dev/null)"; [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= 2>/dev/null | grep -q "_supervise $1"
}

# Códigos de salida: 0 = terminada; 2 = error de configuración (no se reintenta); otro = se reintenta.
run_phase() {
  local phase="$1"; shift
  local envname py
  case "$phase" in
    ollama)
      [ -x "$OLLAMA_BIN" ] || { say "no existe $OLLAMA_BIN"; return 2; }
      export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
      OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=60m "$OLLAMA_BIN" serve ;;
    grobid-server)
      [ -d "$GROBID_DIR" ] || { say "no existe $GROBID_DIR (ver la guía: compilar GROBID 0.8.0)"; return 2; }
      ensure_env grobid || return 2
      export JAVA_HOME; JAVA_HOME="$(env_dir grobid)/lib/jvm"
      export PATH="$JAVA_HOME/bin:$PATH"
      cd "$GROBID_DIR" && ./gradlew --console=plain run ;;
    pdfs|grobid|prune|prepare|ocr|experiment)
      envname=p2mc-nesy; [ "$phase" = ocr ] && envname=p2mc-ocr
      ensure_env "$envname" || return 2
      py="$(env_dir "$envname")/bin/python"
      cd "$REPO" || return 2
      case "$phase" in
        pdfs)       "$py" build_dataset.py pdfs "$@" ;;
        grobid)     need_service grobid-server 8070 900 || return 3
                    "$py" build_dataset.py grobid --workers 4 "$@" ;;
        prune)      "$py" build_dataset.py prune "$@" ;;
        prepare)    export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
                    "$py" experiment_cbr.py prepare --dataset "$DATASET" --force "$@" ;;
        ocr)        export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
                    "$py" build_dataset.py ocr "$@" ;;
        experiment) need_service ollama 11434 120 || return 3
                    export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_XET=1
                    "$py" experiment_cbr.py all --dataset "$DATASET" --generation "$GENERATION" "$@" ;;
      esac ;;
    *) say "fase desconocida: $phase"; return 2 ;;
  esac
}

_supervise() {
  local phase="$1"; shift
  mkdir -p "$STATE/pids"
  echo $$ > "$(pidfile "$phase")"
  local tries=0 max="${P2MC_MAX_RETRIES:-20}" rc
  while :; do
    say "[$phase] arrancando (intento $((tries + 1)))"
    ( run_phase "$phase" "$@" ); rc=$?
    if [ "$rc" -eq 0 ]; then
      say "[$phase] terminado correctamente"; rm -f "$STATE/desired/$phase"; break
    fi
    if [ "$rc" -eq 2 ]; then
      say "[$phase] error de configuración (código 2): no se reintenta"; break
    fi
    tries=$((tries + 1))
    if [ "$tries" -ge "$max" ]; then say "[$phase] $max fallos seguidos: me rindo"; break; fi
    say "[$phase] terminó con código $rc; reintento en ${P2MC_RETRY_WAIT:-30} s"; sleep "${P2MC_RETRY_WAIT:-30}"
  done
  rm -f "$(pidfile "$phase")"
}

cmd_start() {
  local phase="${1:-}"; shift || true
  case " $PHASES " in *" $phase "*) ;; *) die "fase desconocida '$phase'. Fases: $PHASES" ;; esac
  mkdir -p "$STATE/pids" "$STATE/desired" "$LOGS"
  if running "$phase"; then say "$phase ya está en marcha (PID $(cat "$(pidfile "$phase")"))"; return 0; fi
  printf '%s\n' "$*" > "$STATE/desired/$phase"
  nohup setsid bash "$SELF" _supervise "$phase" "$@" >> "$LOGS/$phase.log" 2>&1 < /dev/null &
  echo $! > "$(pidfile "$phase")"
  say "$phase lanzado (PID $!). Log: $LOGS/$phase.log"
}

cmd_resume() {
  mkdir -p "$STATE/desired"
  local p f any=0 args
  for p in $PHASES; do
    f="$STATE/desired/$p"; [ -f "$f" ] || continue
    any=1; args=(); read -r -a args < "$f" || true
    cmd_start "$p" ${args[@]+"${args[@]}"}
  done
  [ "$any" -eq 1 ] || say "no hay fases pendientes (las que terminan bien se quitan solas)"
  sleep 1; cmd_status
}

cmd_stop() {
  local target="${1:-all}" p pid
  for p in $PHASES; do
    [ "$target" = all ] || [ "$target" = "$p" ] || continue
    rm -f "$STATE/desired/$p"
    if running "$p"; then
      pid="$(cat "$(pidfile "$p")")"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      for _ in 1 2 3 4 5 6; do   # espera a que desaparezca el grupo (los zombis no cuentan)
        ps -eo pgid=,stat= | awk -v p="$pid" '$1==p && $2 !~ /^Z/' | grep -q . || break
        sleep 1
      done
      if ps -eo pgid=,stat= | awk -v p="$pid" '$1==p && $2 !~ /^Z/' | grep -q .; then
        say "$p no responde a TERM: lo fuerzo"; kill -KILL -- "-$pid" 2>/dev/null
      fi
      say "$p parado (PID $pid)"
    fi
    rm -f "$(pidfile "$p")"
  done
}

count() { find "$1" -maxdepth 1 -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

cmd_status() {
  local p pid
  echo "== Fases ($(date '+%F %T')) =="
  for p in $PHASES; do
    if running "$p"; then
      pid="$(cat "$(pidfile "$p")")"
      printf '  %-14s EN MARCHA (PID %s, %s)' "$p" "$pid" "$(ps -p "$pid" -o etime= | tr -d ' ')"
    else
      printf '  %-14s parada' "$p"
    fi
    [ -f "$STATE/desired/$p" ] && printf '  [pendiente de terminar: "resume" la relanza]'
    printf '\n'
    [ -f "$LOGS/$p.log" ] && printf '                 último log: %s\n' "$(tail -n 1 "$LOGS/$p.log" | cut -c1-100)"
  done
  if [ -s "$STATE/env" ]; then
    echo "== Variables fijas ($STATE/env) =="; grep -v '^#' "$STATE/env" | sed 's/^/  /'
  fi
  echo "== Servicios =="
  port_open 11434 && echo "  ollama (11434): escuchando" || echo "  ollama (11434): NO"
  port_open 8070  && echo "  grobid (8070):  escuchando" || echo "  grobid (8070):  NO"
  echo "== Entornos en $ROOT/envs =="
  for p in $ENVS; do env_ok "$p" && echo "  $p: ok" || echo "  $p: NO existe (bootstrap lo crea)"; done
  echo "== Datos en $DATA_DIR =="
  echo "  PDFs: $(count "$DATA_DIR/pdfs" '*.pdf')   XML: $(count "$DATA_DIR/xml_outputs" '*.tei.xml')   OCR: $(count "$DATA_DIR/lightocr_jsons" '*.json')"
  local pred="$REPO/results/$DATASET/$GENERATION/predictions/llm.jsonl"
  [ -f "$pred" ] && echo "  predicciones llm: $(wc -l < "$pred")"
  echo "== Disco =="; df -h "$HOME" | tail -1
}

cmd_smoke() {         # prueba de humo con el Ollama real (en primer plano, no supervisada)
  ensure_env p2mc-nesy || die "no se pudo crear el entorno p2mc-nesy"
  need_service ollama 11434 120 || die "Ollama no escucha en el puerto 11434"
  cd "$REPO" && "$(env_dir p2mc-nesy)/bin/python" scripts/smoke_test.py --dataset "$DATASET" --generation "$GENERATION" "$@"
}

cmd_evaluate() {      # evalúa las predicciones guardadas (rápido; se puede hacer con resultados parciales)
  ensure_env p2mc-nesy || die "no se pudo crear el entorno p2mc-nesy"
  cd "$REPO" && "$(env_dir p2mc-nesy)/bin/python" evaluate_predictions.py --dataset "$DATASET" "$@"
}

usage() { sed -n '2,/^set -uo/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; }

case "${1:-help}" in
  bootstrap) shift; for e in ${@:-$ENVS}; do ensure_env "$e" || exit 1; done ;;
  start)     shift; cmd_start "$@" ;;
  resume)    cmd_resume ;;
  stop)      shift; cmd_stop "${1:-all}" ;;
  status)    cmd_status ;;
  smoke)     shift; cmd_smoke "$@" ;;
  evaluate)  shift; cmd_evaluate "$@" ;;
  logs)      tail -n 40 -f "$LOGS/${2:?uso: logs <fase>}.log" ;;
  _supervise) shift; _supervise "$@" ;;
  *)         usage ;;
esac
