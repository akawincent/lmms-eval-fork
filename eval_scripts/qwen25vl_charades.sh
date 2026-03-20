export HOME="/run/determined/workdir/home/"
export HF_HOME="/run/determined/workdir/home/.cache/huggingface"

benchmark=temporal_grounding_charades
output_path=logs
model=Qwen2.5-VL-3B-Instruct
run_name_suffix=_64f
num_gpus=6
num_processes=6
num_frames=64
attn_method=flash_attention_2
device_map=auto

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))
# export CUDA_VISIBLE_DEVICES=1

accelerate launch --num_processes=$num_processes --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/$model,max_pixels=12845056,attn_implementation=$attn_method,interleave_visuals=False,max_num_frames=$num_frames \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_subdir "${model}${run_name_suffix}" \
    --output_path $output_path/$benchmark \
    --use_cache $HOME/db_cache_lmms_eval/charades-sta_qwen25vl_64f.db \
