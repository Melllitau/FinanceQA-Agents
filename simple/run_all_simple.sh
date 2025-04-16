#!/bin/bash
#chmod +x run_all_simple.sh

# Lista de modelos
MODELS=(
  "command-r7b"
  # "gemma3:12b"
  # "gemma3:27b"
  # "deepseek-r1:14b"
  # "deepseek-r1:32b"
  # "qwen2.5:32b"
  # "qwen2.5:14b"
  # "qwq"
)

# Caminho para o seu script Python
SCRIPT="simplemodel.py"

# Loop para cada modelo
for MODEL in "${MODELS[@]}"
do
  # Substitui ":" por "_" para usar no nome do arquivo
  SAFE_NAME=$(echo "$MODEL" | tr ':' '_')
  OUTPUT_TXT="${SAFE_NAME}_simple.txt"
  GPU_LOG="${SAFE_NAME}_gpu_log_simple.csv"
  
  echo "=== Running model: $MODEL ==="
  
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv -lms 500 > "$GPU_LOG" &
  GPU_MONITOR_PID=$!

  # Executa o script principal e mede RAM/CPU
  /usr/bin/time -v python "$SCRIPT" --model "$MODEL" &> "$OUTPUT_TXT"

  # Encerra o modelo do Ollama explicitamente
  ollama stop "$MODEL"
  # Para o monitor de GPU
  kill $GPU_MONITOR_PID

  # Aguarda ele encerrar corretamente
  wait $GPU_MONITOR_PID 2>/dev/null

  # Extrai uso máximo de VRAM
  MAX_VRAM=$(awk -F ',' 'NR > 1 {gsub(/ MiB/, "", $2); if ($2+0 > max) max=$2+0} END {print max}' "$GPU_LOG")

  # Escreve no final do log principal
  {
    echo ""
    echo "Maximum GPU VRAM used (MiB): ${MAX_VRAM}"
  } >> "$OUTPUT_TXT"

  echo "Finalizado: $MODEL"
  echo "Log: $OUTPUT_TXT"
  echo "GPU: $MAX_VRAM MiB"
  echo ""
done
