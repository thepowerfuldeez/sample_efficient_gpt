#!/usr/bin/env bash
# Parallel launcher for facility-location filtering over embeddings.
# - Processes in 15k chunks.
# - Up to 8 concurrent processes.
# Customize DATA/OUT_DIR/END as needed.

# make sure to install submodlib into venv
# uv pip install git+https://github.com/decile-team/submodlib.git

set -euo pipefail

DATA=${DATA:-/path/to/train_embedding.jsonl}
OUT_DIR=${OUT_DIR:-./fla_shards}
CHUNK=${CHUNK:-10000}
MAX_PROCS=${MAX_PROCS:-4}
START=${START:-0}
END=${END:-0} # 0 => auto-detect dataset length inside python

mkdir -p "${OUT_DIR}"

pids=()

launch_chunk() {
  local s=$1
  local e=$2
  local outfile="${OUT_DIR}/fla_select_${s}_${e}.jsonl"
  echo "Launching chunk ${s}-${e} -> ${outfile}"
  python scripts/data_processing/sft/infer_hf_filter_fla.py \
    --data-file "${DATA}" \
    --start "${s}" \
    --end "${e}" \
    --output-file "${outfile}" &
  pids+=($!)
}

wait_for_slots() {
  while (( ${#pids[@]} >= MAX_PROCS )); do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[i]}" 2>/dev/null; then
        wait "${pids[i]}" || true
        unset 'pids[i]'
      fi
    done
    pids=("${pids[@]}") # compact
    sleep 1
  done
}

current=$START
while true; do
  next=$((current + CHUNK))
  # If END==0, let python clip to dataset length; otherwise stop at END
  if (( END > 0 && current >= END )); then
    break
  fi
  if (( END > 0 && next > END )); then
    next=$END
  fi
  wait_for_slots
  launch_chunk "${current}" "${next}"
  current=${next}
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
  wait "${pid}"
done
