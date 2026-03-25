#!/usr/bin/env bash

set -euo pipefail

export HOME="/run/determined/workdir/home/"
export HF_HOME="/run/determined/workdir/home/.cache/huggingface"

task_selector="${1:-${TASK_SELECTOR:-vsisuper_recall}}"
output_path="${OUTPUT_PATH:-logs}"
model="${MODEL:-Cambrian-S-7B-LFP}"
run_name_suffix="${RUN_NAME_SUFFIX:-_memory_design}"
num_gpus="${NUM_GPUS:-7}"
num_processes="${NUM_PROCESSES:-7}"

MIV_TOKEN_LEN="${MIV_TOKEN_LEN:-64}"
SI_TOKEN_LEN="${SI_TOKEN_LEN:-729}"
ENABLE_VISUAL_FEATURE_CACHING="${ENABLE_VISUAL_FEATURE_CACHING:-True}"

declare -a recall_tasks=(
    "vsisuper_recall_10mins"
    "vsisuper_recall_30mins"
    "vsisuper_recall_60mins"
    "vsisuper_recall_120mins"
    "vsisuper_recall_240mins"
)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((num_gpus - 1)))"
else
    export CUDA_VISIBLE_DEVICES
fi

resolve_tasks() {
    case "$1" in
        vsisuper_recall)
            printf "%s\n" "${recall_tasks[@]}"
            ;;
        vsisuper_recall_10mins|vsisuper_recall_30mins|vsisuper_recall_60mins|vsisuper_recall_120mins|vsisuper_recall_240mins)
            printf "%s\n" "$1"
            ;;
        *)
            echo "Unsupported task selector: $1" >&2
            echo "Use vsisuper_recall or one of: ${recall_tasks[*]}" >&2
            exit 1
            ;;
    esac
}

configure_task() {
    case "$1" in
        vsisuper_recall_10mins)
            SENSORY_WINDOW_SIZE=32
            COMPRESSION_DOWNSAMPLE_RATIO=2
            CONSOLIDATION_METHOD=drop_merge
            RETRIEVAL_TOPK=32
            SURPRISE_THRESHOLD=0.35
            CONSOLIDATION_MEM_BUDGET=16384
            ;;
        vsisuper_recall_30mins)
            SENSORY_WINDOW_SIZE=64
            COMPRESSION_DOWNSAMPLE_RATIO=2
            CONSOLIDATION_METHOD=drop
            RETRIEVAL_TOPK=256
            SURPRISE_THRESHOLD=0.3
            CONSOLIDATION_MEM_BUDGET=32768
            ;;
        vsisuper_recall_60mins)
            SENSORY_WINDOW_SIZE=64
            COMPRESSION_DOWNSAMPLE_RATIO=2
            CONSOLIDATION_METHOD=drop
            RETRIEVAL_TOPK=512
            SURPRISE_THRESHOLD=0.25
            CONSOLIDATION_MEM_BUDGET=16384
            ;;
        vsisuper_recall_120mins)
            SENSORY_WINDOW_SIZE=32
            COMPRESSION_DOWNSAMPLE_RATIO=2
            CONSOLIDATION_METHOD=drop_merge
            RETRIEVAL_TOPK=128
            SURPRISE_THRESHOLD=0.25
            CONSOLIDATION_MEM_BUDGET=32768
            ;;
        vsisuper_recall_240mins)
            SENSORY_WINDOW_SIZE=32
            COMPRESSION_DOWNSAMPLE_RATIO=2
            CONSOLIDATION_METHOD=drop
            RETRIEVAL_TOPK=32
            SURPRISE_THRESHOLD=0.35
            CONSOLIDATION_MEM_BUDGET=16384
            ;;
        *)
            echo "No parameter preset for task: $1" >&2
            exit 1
            ;;
    esac
}

run_task() {
    local task="$1"
    configure_task "$task"

    local model_args
    model_args="pretrained=nyu-visionx/${model},conv_template=qwen_2,"
    model_args+="miv_token_len=${MIV_TOKEN_LEN},"
    model_args+="si_token_len=${SI_TOKEN_LEN},"
    model_args+="sensory_window_size=${SENSORY_WINDOW_SIZE},"
    model_args+="compression_downsample_ratio=${COMPRESSION_DOWNSAMPLE_RATIO},"
    model_args+="consolidation_method=${CONSOLIDATION_METHOD},"
    model_args+="retrieval_topk=${RETRIEVAL_TOPK},"
    model_args+="enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING},"
    model_args+="surprise_threshold=${SURPRISE_THRESHOLD},"
    model_args+="consolidation_mem_budget=${CONSOLIDATION_MEM_BUDGET}"

    echo "Running ${task}"
    echo "  sensory_window_size=${SENSORY_WINDOW_SIZE}"
    echo "  consolidation_method=${CONSOLIDATION_METHOD}"
    echo "  retrieval_topk=${RETRIEVAL_TOPK}"
    echo "  surprise_threshold=${SURPRISE_THRESHOLD}"
    echo "  consolidation_mem_budget=${CONSOLIDATION_MEM_BUDGET}"

    accelerate launch --num_processes="${num_processes}" --main_process_port=12346 -m lmms_eval \
        --model cambrians_vsr \
        --force_simple \
        --model_args="${model_args}" \
        --tasks "${task}" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "${model}" \
        --output_subdir "${model}${run_name_suffix}" \
        --output_path "${output_path}/${task}"
}

while IFS= read -r task; do
    run_task "$task"
done < <(resolve_tasks "${task_selector}")
