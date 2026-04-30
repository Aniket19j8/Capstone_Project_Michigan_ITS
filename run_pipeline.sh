#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ITS RAG — One-Shot AWS Pipeline Script
#
# PURPOSE:
#   Spin up the EC2 instance once, run all pipeline steps, upload results to
#   S3, then (optionally) terminate itself. Never needs to stay running.
#
# USAGE (on the EC2 instance after SSH in, or via User Data):
#   chmod +x run_pipeline.sh
#
#   # Full rebuild (new data + fresh vector store):
#   ./run_pipeline.sh --reset
#
#   # Append new dataset to existing index (faster, no re-embedding old data):
#   ./run_pipeline.sh --append --new-source ./data/raw/servicenow --dataset-name servicenow
#
#   # Just rebuild evaluation results without re-embedding:
#   ./run_pipeline.sh --eval-only
#
# REQUIRED ENV VARS (set in /etc/its.env or export before running):
#   S3_BUCKET          e.g. s3://your-its-results-bucket
#   DATABASE_URL       postgresql://user:pass@rds-host:5432/itsdb
#   OLLAMA_BASE_URL    http://localhost:11434  (default)
#
# OPTIONAL ENV VARS:
#   AUTO_SHUTDOWN      set to "true" to terminate the instance after the run
#   MIN_QUALITY        quality filter for preprocessing (default 0.0)
#   BATCH_SIZE         Ollama embedding batch size (default 128)
#   NEW_SOURCE         path to new CSV folder (same as --new-source)
#   DATASET_NAME       label for new source (same as --dataset-name)
#   COLUMN_MAP         JSON column mapping (same as --column-map)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Load env file if present ──────────────────────────────────────────────────
if [ -f /etc/its.env ]; then
  set -a; source /etc/its.env; set +a
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
S3_BUCKET="${S3_BUCKET:-}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-false}"
MIN_QUALITY="${MIN_QUALITY:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NEW_SOURCE="${NEW_SOURCE:-}"
DATASET_NAME="${DATASET_NAME:-custom}"
COLUMN_MAP="${COLUMN_MAP:-{}}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

MODE="rebuild"       # rebuild | append | eval-only
DO_PREPROCESS=true
DO_BUILD=true
DO_EVAL=true
RESET_FLAG=""        # --reset or empty

# ── Parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)        MODE="rebuild"; RESET_FLAG="--reset"; shift ;;
    --append)       MODE="append";  RESET_FLAG="";        shift ;;
    --eval-only)    MODE="eval-only"; DO_PREPROCESS=false; DO_BUILD=false; shift ;;
    --new-source)   NEW_SOURCE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --column-map)   COLUMN_MAP="$2"; shift 2 ;;
    --no-eval)      DO_EVAL=false; shift ;;
    --shutdown)     AUTO_SHUTDOWN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════════════════"
echo "  ITS RAG Pipeline  |  mode: $MODE  |  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════"

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
  source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Verify Ollama is up ───────────────────────────────────────────────────────
echo ""
echo "── Checking Ollama ──────────────────────────────────────"
for i in 1 2 3 4 5; do
  if curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
    echo "  Ollama is up."
    break
  fi
  echo "  Waiting for Ollama ($i/5)..."
  sleep 5
  if [ $i -eq 5 ]; then
    echo "ERROR: Ollama not reachable at $OLLAMA_BASE_URL. Is 'ollama serve' running?"
    exit 1
  fi
done

# Confirm required models are present
for MODEL in qwen3:0.6b qwen3:4b; do
  if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    echo "  Pulling $MODEL ..."
    ollama pull "$MODEL"
  else
    echo "  Model present: $MODEL"
  fi
done

# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
if [ "$DO_PREPROCESS" = true ]; then
  echo ""
  echo "── Step 1: Preprocessing ────────────────────────────────"

  PREPROCESS_ARGS="--min-quality $MIN_QUALITY"

  if [ -n "$NEW_SOURCE" ]; then
    PREPROCESS_ARGS="$PREPROCESS_ARGS --new-source $NEW_SOURCE --dataset-name $DATASET_NAME"
    if [ "$COLUMN_MAP" != "{}" ]; then
      PREPROCESS_ARGS="$PREPROCESS_ARGS --column-map '$COLUMN_MAP'"
    fi
  fi

  if [ "$MODE" = "append" ]; then
    PREPROCESS_ARGS="$PREPROCESS_ARGS --append"
  fi

  # Skip KB reprocessing when only adding tickets in append mode
  if [ "$MODE" = "append" ] && [ -z "$NEW_SOURCE" ]; then
    PREPROCESS_ARGS="$PREPROCESS_ARGS --no-kb"
  fi

  echo "  python 02_preprocess_data.py $PREPROCESS_ARGS"
  eval "python 02_preprocess_data.py $PREPROCESS_ARGS"
fi

# ── Step 2: Build Vector Store ────────────────────────────────────────────────
if [ "$DO_BUILD" = true ]; then
  echo ""
  echo "── Step 2: Build Vector Store ───────────────────────────"

  BUILD_ARGS="--batch-size $BATCH_SIZE $RESET_FLAG"
  if [ "$MODE" = "append" ]; then
    BUILD_ARGS="--batch-size $BATCH_SIZE --append"
  fi

  echo "  python 03_build_vector_store.py $BUILD_ARGS"
  eval "python 03_build_vector_store.py $BUILD_ARGS"
fi

# ── Step 3: Evaluation ────────────────────────────────────────────────────────
if [ "$DO_EVAL" = true ]; then
  echo ""
  echo "── Step 3: Evaluation ───────────────────────────────────"
  python 06_evaluation.py --ablation
  python 11_phase5_evaluation.py --all --sample-size 1000 --retrieval-sample-size 100
fi

# ── Upload results to S3 ──────────────────────────────────────────────────────
if [ -n "$S3_BUCKET" ]; then
  echo ""
  echo "── Uploading results to S3: $S3_BUCKET ─────────────────"

  # Always upload processed data and evaluation output
  aws s3 sync ./data/processed/  "$S3_BUCKET/processed/"  \
    --exclude "*.db" --exclude "*.db-wal" --exclude "*.db-shm" \
    --no-progress
  aws s3 sync ./evaluation/       "$S3_BUCKET/evaluation/" --no-progress

  # Upload ChromaDB only on full rebuild (it can be large — ~5GB at 80k tickets)
  if [ "$MODE" = "rebuild" ]; then
    echo "  Uploading ChromaDB (this may take a few minutes)..."
    aws s3 sync ./data/chroma_db/  "$S3_BUCKET/chroma_db/"  --no-progress
  fi

  # Upload this run's log
  aws s3 cp "$LOG_FILE" "$S3_BUCKET/logs/$LOG_FILE" --no-progress

  echo "  Upload complete."
else
  echo ""
  echo "  ⚠ S3_BUCKET not set — skipping upload. Set S3_BUCKET env var to enable."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline complete  |  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Log: $LOG_FILE"
echo "════════════════════════════════════════════════════════"

# ── Auto-shutdown (terminate the instance to stop billing) ───────────────────
if [ "$AUTO_SHUTDOWN" = "true" ]; then
  echo ""
  echo "AUTO_SHUTDOWN=true — shutting down instance in 60 seconds."
  echo "Kill this script now if you need to stay on."
  sleep 60
  sudo shutdown -h now
fi
