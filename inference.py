import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
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
        # Save-size / quality switch via env var VIDEO_QUALITY (imageio 0-10 scale):
        #   higher = better quality + larger file; lower = smaller. Default 4.8.
        quality = float(os.environ.get("VIDEO_QUALITY", "4.8"))
        imageio.mimwrite(filename, list(video), fps=fps, codec="libx264", quality=quality)
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
import json

from utils.memory_util import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, default=None)
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--name_length", type=int, default=100, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples generated per prompt")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true", help="Report per-sample latency and throughput for the current hardware.")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

# demo_utils.memory.gpu is bound at import time (before set_device), so it is
# cuda:0 on every rank. Re-point it to this rank's device for model placement.
gpu = device
set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    try:
        pipeline.generator.load_state_dict(gen_sd, strict=True)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=True)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    indices = list(range(rank, len(dataset), world_size))
    dataset = Subset(dataset, indices)

dataloader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        # Conditioning chunk = chunk 0, never generated, just VAE-encoded and cached.
        #   framewise (nfpb=1): 1 frame  -> 1 latent
        #   chunkwise (nfpb=3): the first frame replicated to 9 frames -> 3 latents
        # Wan VAE temporal mapping: F frames -> 1 + (F-1)//4 latents, so
        # num_cond_frames = 1 + 4*(nfpb-1) gives exactly nfpb conditioning latents.
        nfpb = config.num_frame_per_block
        num_cond_frames = 1 + 4 * (nfpb - 1)
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image -> [1, C, F, H, W], replicating the first frame for chunkwise
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        if num_cond_frames > 1:
            image = image.repeat(1, 1, num_cond_frames, 1, 1)

        # Encode the input image as the conditioning chunk (nfpb latents)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        assert initial_latent.shape[1] == nfpb, \
            f"expected {nfpb} conditioning latents, got {initial_latent.shape[1]}"
        prompts = [prompt]
        # Per-prompt seed offset: each prompt gets its OWN reproducible noise
        # (seed + global prompt index), independent of GPU count / sharding /
        # skips. Change --seed to sweep a different noise set across all prompts.
        set_seed(args.seed + idx)
        sampled_noise = torch.randn(
            [1, args.num_output_frames - nfpb, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
  
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None

    sample_report_timing = args.report_timing and i >= 1

    for sample_idx in range(args.num_samples):
        file_name = f"{prompt[:args.name_length]}-{sample_idx}.mp4" if args.name_length != -1 else f"{prompt}-{sample_idx}.mp4" 
        output_path = os.path.join(args.output_folder, file_name)

        if os.path.exists(output_path):
            print(f'Video {sample_idx} has been generated. Pass!')
            continue

        sample_seed = args.seed + idx * 1000 + sample_idx
        set_seed(sample_seed)

        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            report_timing=sample_report_timing,
        )

        if sample_report_timing:
            latency = pipeline.first_chunk_time
            elapsed = pipeline.last_generation_time
            num_pixel_frames = video.shape[1]
            fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
            print(f"[Sample {i}] {num_pixel_frames} frames, "
                f"latency {latency:.2f}s, FPS {fps:.2f}")

        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        current_video = (255.0 * current_video).clamp(0, 255).to(torch.uint8)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()
        write_video(output_path, current_video[0], fps=16)
