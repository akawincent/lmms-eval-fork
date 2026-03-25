#!/usr/bin/env bash

set -euo pipefail

export HOME="/run/determined/workdir/home/"
export HF_HOME="/run/determined/workdir/home/.cache/huggingface"

task_selector="${1:-${TASK_SELECTOR:-vsisuper_count_streaming}}"
output_path="${OUTPUT_PATH:-logs}"
model="${MODEL:-Cambrian-S-7B-LFP}"
run_name_suffix="${RUN_NAME_SUFFIX:-_memory_design}"
num_gpus="${NUM_GPUS:-7}"
num_processes="${NUM_PROCESSES:-7}"

MIV_TOKEN_LEN="${MIV_TOKEN_LEN:-64}"
SI_TOKEN_LEN="${SI_TOKEN_LEN:-729}"
ENABLE_VISUAL_FEATURE_CACHING="${ENABLE_VISUAL_FEATURE_CACHING:-True}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:--1}"

declare -a count_streaming_tasks=(
    "vsisuper_count_streaming_10mins"
    "vsisuper_count_streaming_30mins"
    "vsisuper_count_streaming_60mins"
    "vsisuper_count_streaming_120mins"
)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((num_gpus - 1)))"
else
    export CUDA_VISIBLE_DEVICES
fi

resolve_tasks() {
    case "$1" in
        vsisuper_count_streaming)
            printf "%s\n" "${count_streaming_tasks[@]}"
            ;;
        vsisuper_count_streaming_10mins|vsisuper_count_streaming_30mins|vsisuper_count_streaming_60mins|vsisuper_count_streaming_120mins)
            printf "%s\n" "$1"
            ;;
        *)
            echo "Unsupported task selector: $1" >&2
            echo "Use vsisuper_count_streaming or one of: ${count_streaming_tasks[*]}" >&2
            exit 1
            ;;
    esac
}

configure_task() {
    case "$1" in
        vsisuper_count_streaming_10mins|vsisuper_count_streaming_30mins|vsisuper_count_streaming_60mins)
            SENSORY_WINDOW_SIZE=128
            SURPRISE_THRESHOLD=0.39
            ;;
        vsisuper_count_streaming_120mins)
            SENSORY_WINDOW_SIZE=128
            SURPRISE_THRESHOLD=0.41
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
    model_args+="video_max_frames=${VIDEO_MAX_FRAMES},"
    model_args+="miv_token_len=${MIV_TOKEN_LEN},"
    model_args+="si_token_len=${SI_TOKEN_LEN},"
    model_args+="sensory_window_size=${SENSORY_WINDOW_SIZE},"
    model_args+="enable_visual_feature_caching=${ENABLE_VISUAL_FEATURE_CACHING},"
    model_args+="surprise_threshold=${SURPRISE_THRESHOLD}"

    echo "Running ${task}"
    echo "  video_max_frames=${VIDEO_MAX_FRAMES}"
    echo "  sensory_window_size=${SENSORY_WINDOW_SIZE}"
    echo "  surprise_threshold=${SURPRISE_THRESHOLD}"

    accelerate launch --num_processes="${num_processes}" --main_process_port=12348 -m lmms_eval \
        --model cambrians_vsc_streaming \
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
