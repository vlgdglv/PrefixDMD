"""
Entry point: generate VBench evaluation videos with the raw (base) bidirectional
Wan2.1 T2V model (48-step flow-matching sampling, CFG), to serve as the teacher /
upper-bound reference against causal / TF-DMD student outputs.

I/O conventions (dataset loading, distributed sharding, output directory,
per-sample seeding, video writing) mirror `inference.py`; model configuration
(FlowMatchScheduler, timestep_shift=5.0, 48 steps, batched CFG) mirrors the
ODE-trajectory generation script used to build the training set, so the
reference videos come from the same distribution as the DMD targets.

Usage (single GPU):
    python inference_bidirectional_vbench.py \
        --data_path prompts/vbench_all_dimension.txt \
        --output_folder outputs/vbench_bidirectional

Usage (multi-GPU, via torchrun):
    torchrun --standalone --nproc_per_node=8 inference_bidirectional_vbench.py \
        --data_path prompts/vbench_all_dimension.txt \
        --output_folder outputs/vbench_bidirectional
"""
import argparse
import os

import torch
import torch.distributed as dist
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torchvision.io import write_video
except ImportError:
    # torchvision >=0.22 removed torchvision.io.write_video; fall back to imageio.
    import imageio
    import numpy as np

    def write_video(filename, video, fps):
        if hasattr(video, "detach"):
            video = video.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
        else:
            video = np.clip(np.asarray(video), 0, 255).astype(np.uint8)
        quality = float(os.environ.get("VIDEO_QUALITY", "4.8"))
        imageio.mimwrite(filename, list(video), fps=fps, codec="libx264", quality=quality)

from pipeline.bidirectional_vbench_inference import BidirectionalVBenchInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

DEFAULT_DATA_PATH = "prompts/vbench_all_dimension.txt"
DEFAULT_EXTENDED_PROMPT_PATH = "prompts/vbench_all_dimension_extend.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None,
                     help="Optional bidirectional generator checkpoint; omit to use the raw pretrained Wan2.1 weights.")
parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="VBench prompt file, one prompt per line.")
parser.add_argument("--extended_prompt_path", type=str, default="auto",
                     help="Optional VBench-extended prompt file (one per line, aligned with --data_path). "
                          "If given, the EXTENDED text is fed to the model, but output filenames still use "
                          "the original (short) prompt from --data_path, matching inference.py's convention. "
                          "VBench eval now always uses the extended prompts: 'auto' (default) resolves to "
                          f"{DEFAULT_EXTENDED_PROMPT_PATH} whenever --data_path is left at its default, and to "
                          "no extended prompts otherwise; pass an explicit path to force extension with a "
                          "custom --data_path, or '' to disable it entirely.")
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--name_length", type=int, default=-1, help="Truncate prompt to this many chars for the output filename. -1 = full prompt (VBench's own matching expects the full prompt text as filename prefix).")
parser.add_argument("--model_name", type=str, default="Wan2.1-T2V-1.3B")
parser.add_argument("--timestep_shift", type=float, default=5.0)
parser.add_argument("--num_inference_steps", type=int, default=48)
parser.add_argument("--guidance_scale", type=float, default=6.0)
parser.add_argument("--context_noise", type=float, default=0)
parser.add_argument("--num_samples", type=int, default=5, help="VBench's standard protocol asks for 5 videos per prompt.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--num_decode_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--frame_per_chunk", type=int, default=3, help="Number of overlap frames between sliding windows")
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                     help="float32 matches the ODE-trajectory generation script this pipeline mirrors; "
                          "bfloat16 is faster/lighter but was not what the teacher trajectories were sampled in.")
parser.add_argument("--contexted_chunk_infer", action="store_true")
parser.add_argument(
    "--dimension_prompt_path", type=str, default=None,
    help="Optional path to a single-dimension VBench prompt subset file (one prompt per line). When given, "
         "the full --data_path dataset is pruned down to only the prompts that also appear in this subset "
         "(matched by exact prompt text); only that dimension's videos are generated. The inference loop "
         "itself is unchanged -- this only trims which prompts the dataset yields.",
)
args = parser.parse_args()
if args.extended_prompt_path == "auto":
    args.extended_prompt_path = DEFAULT_EXTENDED_PROMPT_PATH if args.data_path == DEFAULT_DATA_PATH else None
elif not args.extended_prompt_path:  # explicit "" disables it
    args.extended_prompt_path = None

# Initialize distributed inference (same convention as inference.py).
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda")
    local_rank = 0

set_seed(args.seed)
torch.set_grad_enabled(False)

pipeline = BidirectionalVBenchInferencePipeline(
    device=device,
    model_name=args.model_name,
    timestep_shift=args.timestep_shift,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    latent_shape=(args.num_output_frames, 16, 60, 104),
)
if args.checkpoint_path:
    pipeline.load_generator_checkpoint(args.checkpoint_path, use_ema=args.use_ema, key="critic")

pipeline = pipeline.to(device=device, dtype=getattr(torch, args.dtype))
pipeline.generator.eval()
pipeline.text_encoder.eval()
pipeline.vae.eval()

dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
print(f"Number of prompts: {len(dataset)}")

if args.dimension_prompt_path and args.dimension_prompt_path != "" :
    with open(args.dimension_prompt_path, encoding="utf-8") as f:
        dimension_prompts = {line.rstrip() for line in f}
    matched_indices = [i for i, p in enumerate(dataset.prompt_list) if p in dimension_prompts]
    unmatched = dimension_prompts - {dataset.prompt_list[i] for i in matched_indices}
    if unmatched:
        print(f"[WARN] {len(unmatched)} prompt(s) in --dimension_prompt_path were not found in --data_path (exact-text match).")
    print(f"[dimension_prompt_path] {args.dimension_prompt_path}: pruned dataset {len(dataset)} -> {len(matched_indices)} prompts")
    dataset = Subset(dataset, matched_indices)

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    dataset = Subset(dataset, indices)

dataloader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()

for batch in tqdm(dataloader, disable=(local_rank != 0)):
    idx = batch["idx"].item()
    prompt = batch["prompts"][0]  # used for the output filename
    extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
    model_prompt = extended_prompt if extended_prompt is not None else prompt  # used as model input text

    for sample_idx in range(args.num_samples):
        file_name = (
            f"{prompt[:args.name_length]}-{sample_idx}.mp4" if args.name_length != -1
            else f"{prompt}-{sample_idx}.mp4"
        )
        output_path = os.path.join(args.output_folder, file_name)

        if os.path.exists(output_path):
            print(f"Video {sample_idx} for prompt {idx} has been generated. Pass!")
            continue

        sample_seed = args.seed + idx * 1000 + sample_idx
        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)
        noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device, dtype=torch.bfloat16, generator=generator,
        )
        if args.contexted_chunk_infer:
            context_latent = torch.load(f"outputs/visual_check/wan2.1_contexted/full_21_latent_{idx}.pt")
            video = pipeline.inference_contexted_chunk(
                text_prompts=[model_prompt],
                generator=generator,
                context_latent=context_latent.to(device=device, dtype=torch.bfloat16),
                show_progress=(local_rank == 0),
                frame_num=args.num_output_frames,
                frame_per_chunk=args.frame_per_chunk,
                context_noise=args.context_noise,
            )
            current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            current_video = (255.0 * current_video).clamp(0, 255).to(torch.uint8)
            write_video( os.path.join(args.output_folder, f"chunked_infer_{idx}.mp4"), current_video[0], fps=args.fps)
        else: 
            video, latent = pipeline.inference(
                text_prompts=[model_prompt],
                generator=generator,
                noise=noise,
                show_progress=(local_rank == 0),
                return_latents=True
            )
    
            torch.save(latent.cpu(), os.path.join(args.output_folder, f"full_21_latent_{idx}.pt"))
            current_video = video[:, :args.num_decode_frames, ...]
            current_video = rearrange(current_video, "b t c h w -> b t h w c").cpu()
            current_video = (255.0 * current_video).clamp(0, 255).to(torch.uint8)
            write_video(output_path, current_video[0], fps=args.fps)

if dist.is_initialized():
    dist.barrier()
