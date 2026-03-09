# Qwen2.5-VL Notes

Paper: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)

## Core video-design takeaway

Qwen2.5-VL does not use the Qwen3-VL style of inserting textual timestamps like `<3.0 seconds>` into the prompt.

Instead, it improves video time modeling by:

- dynamic FPS training
- absolute-time-aligned MRoPE
- computing temporal position intervals from the effective video sampling rate

The paper is explicit about this:

- it introduces "dynamic frame rate (FPS) training and absolute time encoding"
- it says Qwen2.5-VL aligns MRoPE IDs directly with timestamps
- it contrasts this with approaches that use textual timestamps or extra grounding heads

Relevant paper passages:

- `colm2024_conference.tex`: "dynamic FPS sampling" and "MRoPE aligns time IDs with absolute time"
- `colm2024_conference.tex`: "Unlike other approaches that incorporate textual timestamps ... we align MRoPE IDs directly with the timestamps."
- `colm2024_conference.tex`: Qwen2.5-VL fixes Qwen2-VL's issue where temporal IDs were tied only to frame count instead of absolute time / content pace

## HF transformers implementation

In `transformers`, Qwen2.5-VL time information is carried through `fps`, not `video_metadata`.

### Processor path

`Qwen2_5_VLProcessor.__call__` takes:

- `images`
- `text`
- `videos`
- `**kwargs`

It does not expose a `video_metadata=` argument.

Inside `processing_qwen2_5_vl.py`:

- it reads `fps = output_kwargs["videos_kwargs"].get("fps", 2.0)`
- it computes `second_per_grid_ts = temporal_patch_size / fps`
- it returns `second_per_grid_ts` together with `pixel_values_videos` and `video_grid_thw`

### Model path

`Qwen2_5_VLForConditionalGeneration.forward` accepts:

- `pixel_values_videos`
- `video_grid_thw`
- `second_per_grid_ts`

Inside `get_rope_index(...)`:

- video temporal position intervals are derived from `second_per_grid_ts`
- if `second_per_grid_ts` is missing, it falls back to `1.0`

So the real temporal-alignment input for Qwen2.5-VL is:

`fps -> second_per_grid_ts -> absolute-time-aligned temporal MRoPE`

not:

`video_metadata -> textual timestamps`

## qwen-vl-utils compatibility detail

`qwen_vl_utils.process_vision_info(...)` has a compatibility branch for Qwen2.5-VL:

- when `return_video_metadata=False`
- it adds sampled `fps` into returned `video_kwargs`

This is a strong hint about the intended integration pattern for Qwen2.5-VL:

- no need to pass `video_metadata`
- but you should preserve and pass `video_kwargs` / sampled `fps`

## Implication for lmms-eval

For `lmms_eval/models/chat/qwen2_5_vl.py` and `lmms_eval/models/simple/qwen2_5_vl.py`:

- adding `video_metadata=...` to `self.processor(...)` is not the right fix
- the processor does not use that interface
- the more relevant fix is to pass `**video_kwargs` (especially `fps`) into the processor call

Otherwise, Qwen2.5-VL may still lose correct temporal scaling and build `second_per_grid_ts` from the default `fps=2.0`.
