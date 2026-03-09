# Qwen3-VL Paper, Repo, and `transformers` Forward Notes

## Scope

This note combines three sources:

1. Paper: `arXiv:2511.21631` (`Qwen3-VL Technical Report`)
2. Official repo: `QwenLM/Qwen3-VL`
3. Hugging Face `transformers==4.57.1` implementation of `Qwen3-VL`

The goal is to answer one concrete question: what does Qwen3-VL claim architecturally, where is that reflected in the public repo, and how does the actual forward path work in code.

## Executive Summary

Qwen3-VL is not just "Qwen2.5-VL but bigger". The paper makes three main architectural claims:

1. Interleaved-MRoPE replaces the earlier axis-partitioned MRoPE layout to improve long video modeling.
2. DeepStack injects intermediate ViT features into the first few decoder layers instead of relying only on the final visual tokens.
3. Video time is no longer encoded mainly through temporal RoPE offsets; instead, timestamps are inserted as text tokens in the prompt stream.

The important result after reading the `transformers` code is that all three claims are real and visible in the implementation:

1. DeepStack is explicitly implemented in the vision encoder and injected into the text decoder.
2. Video timestamps are literally rendered into the prompt as strings like `<3.0 seconds>`.
3. The multimodal forward path is "processor expands placeholders -> vision encoder produces visual embeddings + deepstack features -> model replaces image/video placeholder token embeddings -> text decoder consumes the fused sequence".

For this `lmms-eval` fork, there is already a Qwen3-VL backend, but the simple backend currently does not pass video metadata back into the processor. That matters because Qwen3-VL's official processor uses video FPS metadata to generate timestamp text, which is one of the paper's explicit design changes.

## What The Paper Actually Adds

## 1. Architecture

The paper describes a three-module model:

1. Vision encoder: SigLIP-2 based ViT with dynamic resolution support
2. Vision-language merger: an MLP merger that compresses visual patches into LLM-width tokens
3. Language model: Qwen3 text backbone, available in dense and MoE variants

The paper's distinctive additions are:

1. Interleaved-MRoPE
2. DeepStack
3. Text-based timestamps for video

These are not presentation-only ideas. They change how sequence positions and visual features are represented before and inside the decoder.

## 2. Training Story

The pretraining recipe is staged:

1. S0: merger-only alignment at 8K
2. S1: full multimodal pretraining at 8K
3. S2: full long-context pretraining at 32K
4. S3: ultra-long adaptation at 256K

The post-training recipe is also staged:

1. SFT
2. strong-to-weak distillation
3. RL

The paper also emphasizes a data change, not just a model change:

1. stronger OCR and document parsing
2. much more long-document and long-video data
3. grounding, spatial reasoning, 3D grounding, GUI agent, and multimodal coding data
4. explicit "thinking with images" post-training

This matters for evaluation: many of the gains are not from a single clever block, but from pushing video/document/agent data harder while preserving text capability.

## 3. Claimed Capability Highlights

From the paper tables and sections:

1. Qwen3-VL-235B-A22B is especially strong on MathVista-mini, MathVision, MathVerse-mini, LogicVista, VisuLogic, document OCR, long-doc understanding, grounding, spatial tasks, and GUI-agent tasks.
2. Long-context video is a central target, not a side effect. The paper reports perfect accuracy up to 30 minutes on its video needle-in-a-haystack setup at native 256K context, and still very high accuracy with YaRN extension to about 1M tokens.
3. Smaller models matter too. The paper repeatedly stresses that 8B and 32B variants inherit a surprising amount of the flagship's capability.

## What The Official Repo Contains

After reading the official `QwenLM/Qwen3-VL` repo docs, the repo is best understood as an integration and usage repo, not the canonical source of the core model forward implementation.

What is in the repo:

1. the main README
2. cookbooks
3. deployment examples
4. evaluation scripts
5. `qwen-vl-utils`

What is not really in the repo:

1. the authoritative decoder/vision forward implementation used by `transformers`

That is why the repo README repeatedly points users to:

1. `transformers>=4.57.0`
2. `AutoModelForImageTextToText`
3. `AutoProcessor`
4. `qwen-vl-utils==0.0.14`

So the repo is the operational layer, while the actual modeling logic lives upstream in Hugging Face.

## Repo-Level Usage Pattern

The repo's canonical inference pattern is:

1. build HF-style `messages`
2. call `processor.apply_chat_template(...)`
3. optionally use `qwen_vl_utils.process_vision_info(...)`
4. call `processor(...)` to build:
   - `input_ids`
   - `pixel_values`
   - `pixel_values_videos`
   - `image_grid_thw`
   - `video_grid_thw`
   - sometimes `video_metadata`
5. call `model.generate(...)`

The repo also highlights two practical points that match the paper:

1. separate image and video pixel budgets
2. timestamp-aware video handling through the processor and `qwen-vl-utils`

## `transformers` Forward Path

The real implementation is in:

`transformers/models/qwen3_vl/modeling_qwen3_vl.py`

and the processor logic is in:

`transformers/models/qwen3_vl/processing_qwen3_vl.py`

Below is the actual end-to-end path.

## 1. Processor Stage

The processor is not a thin wrapper. It performs a major part of the multimodal assembly.

### Images

For images, the processor:

1. runs the image processor
2. gets `image_grid_thw`
3. replaces each image placeholder in the prompt with the correct count of image tokens

The number of placeholder tokens is derived from the merged visual grid, not from a fixed constant.

### Videos

For videos, the processor does more work:

1. runs the video processor
2. gets `video_grid_thw`
3. keeps or pops `video_metadata`
4. computes timestamps from frame indices and FPS
5. rewrites each video placeholder into repeated timestamped frame groups

The key implementation detail is this:

1. each frame group becomes something like `<t seconds><vision_start><video_token ...><vision_end>`
2. timestamps are computed from sampled frame indices and video FPS
3. timestamps are averaged per temporal merge group

So "text-based timestamp alignment" is literally implemented by injecting timestamp strings into the tokenized prompt before the model forward.

## 2. Vision Encoder Path

Inside `Qwen3VLVisionModel.forward`:

1. `patch_embed` converts raw visual patches into hidden states
2. interpolated absolute position embeddings are added
3. rotary positional embeddings are built from `grid_thw`
4. ViT blocks run over the visual sequence
5. selected intermediate layers are routed through `deepstack_merger_list`
6. final visual states go through `self.merger`

So the vision encoder returns two products:

1. final merged visual embeddings
2. a list of DeepStack visual embeddings from intermediate ViT layers

This is the paper's DeepStack claim made concrete.

## 3. Placeholder Replacement

Inside `Qwen3VLModel.forward`:

1. text token embeddings are created first
2. image/video features are computed by `get_image_features` / `get_video_features`
3. placeholder masks are built from `input_ids`
4. image/video embeddings are inserted into the text embedding sequence with `masked_scatter`

This means the decoder ultimately sees one fused token stream, where visual placeholders have already been replaced by learned visual embeddings.

## 4. DeepStack Injection

After placeholder replacement, the model also prepares:

1. `visual_pos_masks`
2. `deepstack_visual_embeds`

Then `Qwen3VLTextModel.forward` runs decoder layers. After each of the first few layers, it conditionally calls `_deepstack_process`, which:

1. selects positions corresponding to visual tokens
2. adds the corresponding DeepStack features into those hidden states

This is the exact code realization of:

1. "inject visual tokens from intermediate ViT layers"
2. "fuse them into early LLM layers"

So Qwen3-VL has two visual injection paths:

1. normal visual token replacement before decoding
2. DeepStack residual feature injection during early decoding

## 5. RoPE / Position Logic

The most important position function is `get_rope_index`.

For images and videos, it builds 3-axis position IDs for:

1. temporal
2. height
3. width

But for video there is a notable change. The code explicitly repeats and splits `video_grid_thw` so that each temporal group is handled separately, then sets temporal grid usage so that:

1. video temporal patches use textual timestamps
2. the temporal index used for RoPE is effectively flattened to 0 per visual frame group

The code comment makes the intent clear: Qwen3-VL uses timestamps rather than absolute-time position IDs.

This is one of the clearest paper-to-code correspondences in the whole model.

## 6. Generation Path

`Qwen3VLForConditionalGeneration.forward` is thin:

1. call `self.model(...)`
2. project hidden states with `lm_head`
3. optionally compute loss

For generation, `prepare_inputs_for_generation` avoids resending image/video tensors after the prefill step:

1. first step uses visual inputs
2. later decode steps reuse KV cache and omit `pixel_values` / `pixel_values_videos`

That is the standard autoregressive optimization, but it matters for tracing actual runtime behavior.

## Paper-to-Code Mapping

### Interleaved-MRoPE

Paper claim:

1. spread temporal, height, and width frequencies more evenly for better long-video behavior

Code evidence:

1. Qwen3-VL config validates `mrope_interleaved`
2. multimodal position IDs are 3-axis
3. video temporal position is no longer mainly carried by absolute temporal RoPE ids
4. explicit timestamp text is injected before tokenization

Interpretation:

The effective time representation is now split between:

1. textual timestamps in the prompt
2. spatial multimodal RoPE for visual token structure

### DeepStack

Paper claim:

1. use intermediate ViT features, not just the final merged visual tokens

Code evidence:

1. `deepstack_visual_indexes = [8, 16, 24]` in config defaults
2. `Qwen3VLVisionModel` collects intermediate features
3. `Qwen3VLTextModel.forward` injects them into early decoder layers

Interpretation:

This is not an auxiliary head or training-only trick. It is part of the inference-time forward graph.

### Text Timestamp Alignment

Paper claim:

1. replace the old temporal-position-heavy design with explicit textual timestamps

Code evidence:

1. processor calculates timestamps from FPS and frame indices
2. video placeholders are expanded to repeated `<x.x seconds>` strings plus visual spans
3. `get_rope_index` comment says Qwen3-VL uses timestamps rather than absolute time position ids

Interpretation:

The processor is an architectural component here, not just preprocessing sugar.

## What This Means For `lmms-eval`

This repo already has Qwen3-VL wrappers:

1. `lmms_eval/models/simple/qwen3_vl.py`
2. `lmms_eval/models/chat/qwen3_vl.py`

That is useful, because it shows how Qwen3-VL is currently wired into evaluation.

## Good News

The wrappers already use the right broad strategy:

1. `AutoProcessor`
2. `process_vision_info(...)`
3. `apply_chat_template(...)`
4. `model.generate(...)`

That matches the official repo and HF implementation.

## Important Caveat

The chat wrapper is closer to the official usage than the simple wrapper.

### Chat wrapper

`lmms_eval/models/chat/qwen3_vl.py` does the right thing for video:

1. `return_video_kwargs=True`
2. `return_video_metadata=True`
3. passes `video_metadata=...` into the processor

That matches the official Qwen3-VL flow.

### Simple wrapper

`lmms_eval/models/simple/qwen3_vl.py` currently does:

1. `return_video_kwargs=False`
2. `return_video_metadata=False`
3. does not pass `video_metadata` into the processor

This matters because `processing_qwen3_vl.py` explicitly warns that Qwen3-VL needs FPS metadata to construct prompt timestamps accurately. When metadata is absent, the processor falls back to `fps=24`.

### Practical consequence

For image-only tasks, this likely does not matter.

For video tasks, especially:

1. temporal grounding
2. long-video QA
3. timestamp-sensitive evaluation

the simple wrapper can diverge from the paper's intended behavior and the official recipe.

## Bottom Line

If the question is "what is the true forward process of Qwen3-VL?", the answer is:

1. prompts are expanded by the processor into multimodal token placeholders
2. video placeholders become timestamped textual spans plus visual spans
3. the vision encoder produces final visual tokens and DeepStack intermediate features
4. the model replaces placeholder token embeddings with visual embeddings
5. the decoder runs with multimodal RoPE and early-layer DeepStack injection
6. generation proceeds autoregressively with visual inputs only in prefill

If the question is "what should I remember for this repo?", the answer is:

1. the official Qwen3-VL behavior depends on processor-side timestamp construction
2. the chat backend in this fork preserves that better than the simple backend
3. any serious video evaluation should verify that `video_metadata` and video kwargs are preserved end to end

## Source Pointers

Paper source unpacked on the remote dev box:

1. `~/.cache/nanochat/knowledge/2511.21631/colm2024_conference.tex`

Key Hugging Face files inspected on the remote dev box:

1. `.../site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
2. `.../site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py`
3. `.../site-packages/transformers/models/qwen3_vl/configuration_qwen3_vl.py`

Relevant files in this repo:

1. `lmms_eval/models/simple/qwen3_vl.py`
2. `lmms_eval/models/chat/qwen3_vl.py`
