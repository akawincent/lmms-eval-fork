export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

benchmark=vsibench
output_path=logs
root_path=/run/determined/workdir/home/
model=LLaVA-OneVision-1.5-4B-Instruct
run_name_suffix=_32f
num_gpus=8
num_processes=8
num_frames=32
attn_method=flash_attention_2
device_map=auto

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))

accelerate launch --num_processes=$num_processes --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=lmms-lab/$model,attn_implementation=$attn_method,max_pixels=3240000,max_num_frames=$num_frames \
    --tasks=$benchmark \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_subdir "${model}${run_name_suffix}" \
    --output_path $output_path/$benchmark \
    --use_cache $root_path/db_cache_lmms_eval/vsibench_llava-ov-1.5.db \