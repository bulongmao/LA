#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LocateAnything-3B LoRA training script for food-plate task
# V100 compatible version
# ============================================================

# -------------------------
# 0. Project paths
# -------------------------
export EAGLE_EMBODIED_ROOT=${EAGLE_EMBODIED_ROOT:-"/data/ljy/locate_anything_project/Eagle/Embodied"}
cd "$EAGLE_EMBODIED_ROOT"

export PYTHONPATH="$EAGLE_EMBODIED_ROOT:${PYTHONPATH:-}"

# -------------------------
# 1. CUDA / NCCL / HF env
# -------------------------
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.2"}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME=${HF_HOME:-"/data/ljy/huggingface_home"}
unset TRANSFORMERS_CACHE

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

# -------------------------
# 2. Distributed settings
# -------------------------
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# -------------------------
# 3. Model / data paths
# -------------------------
MODEL_PATH=${MODEL_PATH:-"/data/ljy/locate_anything_project/models/LocateAnything-3B"}

META_PATH=${META_PATH:-"/data/ljy/locate_anything_project/lora_food_plate_data/meta_train_only_jsonl.json"}

if [[ ! -f "$META_PATH" ]]; then
  echo "[ERROR] META_PATH not found: $META_PATH" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2
  exit 1
fi

# -------------------------
# 4. Training hyperparameters
# -------------------------
MAX_STEPS=${MAX_STEPS:-3000}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}

LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-300}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}

# 注意：
# 你当前 zero_stage1_config.json 里之前实际是 gradient_accumulation_steps=4，
# 所以这里默认设成 4，避免和 DeepSpeed 配置冲突。
GRADIENT_ACC=${GRADIENT_ACC:-4}

MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-2048}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-2048}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config.json"}

# -------------------------
# 5. LoRA settings
# -------------------------
USE_LLM_LORA=${USE_LLM_LORA:-64}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}

FREEZE_LLM=${FREEZE_LLM:-True}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-True}
FREEZE_MLP=${FREEZE_MLP:-False}

# -------------------------
# 6. Output dir
# -------------------------
# 默认新建一个带步数和时间戳的目录，避免误用旧 checkpoint 导致“不跑了”
RUN_TAG=${RUN_TAG:-"food_plate_lora_v100_${MAX_STEPS}step_$(date +%Y%m%d_%H%M%S)"}

OUTPUT_DIR=${OUTPUT_DIR:-"/data/ljy/locate_anything_project/lora_foutputs/${RUN_TAG}"}

mkdir -p "$OUTPUT_DIR"

script_name=$(basename "${BASH_SOURCE[0]}")

echo "============================================================"
echo "[INFO] Start LocateAnything LoRA training"
echo "============================================================"
echo "[INFO] EAGLE_EMBODIED_ROOT: $EAGLE_EMBODIED_ROOT"
echo "[INFO] MODEL_PATH:          $MODEL_PATH"
echo "[INFO] META_PATH:           $META_PATH"
echo "[INFO] OUTPUT_DIR:          $OUTPUT_DIR"
echo "[INFO] GPUS:                $GPUS"
echo "[INFO] MAX_STEPS:           $MAX_STEPS"
echo "[INFO] SAVE_STEPS:          $SAVE_STEPS"
echo "[INFO] SAVE_TOTAL_LIMIT:    $SAVE_TOTAL_LIMIT"
echo "[INFO] LR:                  $LR"
echo "[INFO] WARMUP_STEPS:        $WARMUP_STEPS"
echo "[INFO] MAX_SEQ_LENGTH:      $MAX_SEQ_LENGTH"
echo "[INFO] GRADIENT_ACC:        $GRADIENT_ACC"
echo "============================================================"

# -------------------------
# 7. Launch training
# -------------------------
LAUNCHER=pytorch python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation eager \
  --causal_attn False \
  --freeze_llm "$FREEZE_LLM" \
  --freeze_mlp "$FREEZE_MLP" \
  --freeze_backbone "$FREEZE_BACKBONE" \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora "$USE_BACKBONE_LORA" \
  --vision_select_layer -1 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 False \
  --fp16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --video_total_pixels 8192 \
  --sample_log_interval 1 \
  --packing_buffer_size "$PACKING_BUFFER_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_NUM_TOKENS_PER_SAMPLE" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --report_to "tensorboard" \
  --run_name "$script_name" \
  --use_onelogger False \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"