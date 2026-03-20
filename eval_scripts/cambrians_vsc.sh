export HOME="/run/determined/workdir/home/"
export HF_HOME="/run/determined/workdir/home/.cache/huggingface"

benchmark=vsisuper_vsc_10mins
output_path=logs
model=Cambrian-S-7B
run_name_suffix=_128f
num_gpus=7
num_processes=7
attn_method=flash_attention_2
device_map=auto

NUM_FRAMES=128
MIV_TOKEN_LEN=64
SI_TOKEN_LEN=729

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))

accelerate launch --num_processes=$num_processes --main_process_port=12346 -m lmms_eval \
    --model cambrians \
    --force_simple \
    --model_args=pretrained=nyu-visionx/$model,conv_template=qwen_2,video_max_frames=${NUM_FRAMES},miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN} \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_subdir "${model}${run_name_suffix}" \
    --output_path $output_path/$benchmark \
