"""
SDEdit-style noise-injection probe: given an existing video latent (e.g. from
another model's output, or the VAE encoding of a real video) with the same
shape as the bidirectional Wan2.1 T2V generator's noise input, add noise up
to a partial strength t* in (0, 1), then denoise from t* back to 0 along the
NORMAL (full, unmodified) Wan sampling schedule using the raw/base
bidirectional Wan 1.3B model (not causal, no KV cache -- every frame attends
to every other frame every step, same as
`pipeline/bidirectional_vbench_inference.py`).

Purpose: inspect what Wan actually changes about an input video at several
edit strengths. Low t* (e.g. 0.3) should only touch fine texture/detail;
high t* (e.g. 0.7) can restructure motion/content. Comparing the outputs
across t* -- and against the un-edited input -- localizes how aggressively
Wan's prior overrides vs. preserves the given video.

t* semantics: `scheduler.timesteps` (after `FlowMatchScheduler.set_timesteps`)
are the WARPED diffusion timesteps in [0, 1000] used throughout this repo's
training/inference code (see `utils/scheduler.py`, `pipeline/bidirectional_vbench_inference.py`).
t* directly selects the target timestep as `t* * 1000` on that normal,
already-shift-warped schedule, then takes the nearest step on it and the
tail from there to 0 -- i.e. "the normal schedule, started partway through"
exactly as requested, not a schedule recomputed over a smaller range (that
would be `FlowMatchScheduler`'s own `denoising_strength` argument, which
this script deliberately does NOT use, to keep the exact same step spacing/
density as full unconditional generation).

Usage:
    python pipeline/sdedit_strength_probe.py \
        --input_latent_path path/to/some_latent.pt \
        --prompt "a cat playing piano" \
        --output_folder outputs/sdedit_probe \
        --t_star 0.3 0.5 0.7

If the .pt file already stores its own prompt (e.g. the
`utils/dataset.py:ODERegressionPTDataset` format, with a top-level "prompt"
key, or an ODE-trajectory "states" entry), --prompt can be omitted.
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from tqdm import tqdm

try:
    from torchvision.io import write_video
except ImportError:
    # torchvision >=0.22 removed torchvision.io.write_video; fall back to imageio.
    import imageio
    import numpy as np

    def write_video(filename, video, fps):
        video = video.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
        quality = float(os.environ.get("VIDEO_QUALITY", "4.8"))
        imageio.mimwrite(filename, list(video), fps=fps, codec="libx264", quality=quality)

from pipeline.bidirectional_vbench_inference import BidirectionalVBenchInferencePipeline
from utils.misc import set_seed


class SDEditStrengthPipeline(BidirectionalVBenchInferencePipeline):
    """
    Adds a single method to the raw bidirectional Wan2.1 pipeline: noise an
    existing latent to a partial strength t* and denoise it back to 0 along
    the tail of the normal full sampling schedule. Everything else (model
    loading, CFG, dtype handling, VAE decode) is inherited unchanged from
    `BidirectionalVBenchInferencePipeline`.
    """

    @torch.no_grad()
    def denoise_from_strength(
        self,
        latent: torch.Tensor,
        text_prompts: List[str],
        t_star: float,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Inputs:
            latent: [B, F, C, H, W] existing (clean-ish) latent, same shape
                as this pipeline's noise input.
            t_star: target timestep as a fraction of [0, 1000] on the NORMAL
                (full) warped schedule, e.g. 0.3 -> timestep ~300. The
                nearest step on `self.scheduler.timesteps` is used as the
                actual noising/starting point; denoising then runs the tail
                of the normal schedule from there down to 0.
        Outputs:
            video: [B, F, H, W, 3] in [0, 1].
            latents: final denoised latent, same shape as `latent`.
            info: dict with the resolved starting timestep / step count, for
                logging.
        """
        assert 0.0 < t_star <= 1.0, f"t_star must be in (0, 1], got {t_star}"
        batch_size = latent.shape[0]
        latent = latent.to(device=self.device, dtype=self.dtype)

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = {
            k: v.repeat(batch_size, *([1] * (v.ndim - 1))) for k, v in self.unconditional_dict.items()
        }

        # `self.scheduler.timesteps` is the NORMAL full schedule (built once
        # in __init__ with denoising_strength=1.0) -- deliberately not
        # recomputed here, so step spacing/density matches full generation.
        full_timesteps = self.scheduler.timesteps
        target_timestep = t_star * 1000.0
        start_idx = int(torch.argmin((full_timesteps - target_timestep).abs()).item())
        start_timestep_value = full_timesteps[start_idx].item()
        run_timesteps = full_timesteps[start_idx:]

        noise = torch.randn(latent.shape, device=self.device, dtype=self.dtype, generator=generator)
        start_timestep_tensor = torch.full(
            [batch_size * self.num_frames], start_timestep_value, device=self.device, dtype=torch.float32,
        )
        latents = self.scheduler.add_noise(
            latent.flatten(0, 1), noise.flatten(0, 1), start_timestep_tensor,
        ).unflatten(0, latent.shape[:2]).to(self.dtype)

        for t in tqdm(run_timesteps, disable=not show_progress, desc=f"t*={t_star}"):
            timestep = t * torch.ones([batch_size, self.num_frames], device=self.device, dtype=torch.float32)

            batched_latents = torch.cat([latents, latents], dim=0)
            batched_timestep = torch.cat([timestep, timestep], dim=0)
            batched_conditional_dict = {
                k: torch.cat([unconditional_dict[k], conditional_dict[k]], dim=0)
                for k in conditional_dict
            }

            flow_pred_batched, _ = self.generator(
                noisy_image_or_video=batched_latents,
                conditional_dict=batched_conditional_dict,
                timestep=batched_timestep,
            )
            flow_pred_uncond, flow_pred_cond = flow_pred_batched.chunk(2, dim=0)
            flow_pred = flow_pred_uncond + self.guidance_scale * (flow_pred_cond - flow_pred_uncond)

            latents = self.scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                latents.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
            # `.step()` has no internal dtype cast; under mixed precision this
            # silently upcasts to float32, breaking the next forward pass
            # against lower-precision weights. Force it back (same fix as
            # the base pipeline's `inference()`).
            latents = latents.to(self.dtype)

        video = self.vae.decode_to_pixel(latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        info = {
            "t_star": t_star,
            "start_timestep": start_timestep_value,
            "num_denoising_steps": len(run_timesteps),
            "total_schedule_steps": len(full_timesteps),
        }
        return video, latents, info


def load_input_latent(path: str, latent_shape) -> Tuple[torch.Tensor, Optional[str]]:
    """
    Loads a video latent from a .pt file, tolerating a few container formats
    since the latent may come from this repo's own ODE-trajectory dumps, a
    different model's raw output, or a VAE encoding done ad hoc elsewhere:
        - a bare torch.Tensor
        - a dict with "states" (ODE-trajectory format, see
          `utils/dataset.py:ODERegressionPTDataset` / `load_pt_sample` in
          `pipeline/teacher_forcing_rollout_probe.py`): the LAST entry along
          the trajectory dimension (the clean endpoint) is used.
        - a dict with "latent" / "latents" / "clean_latent" / "video_latent"
    A leading singleton batch dim is squeezed/added as needed so the result
    is always [1, F, C, H, W]. The dict's "prompt" field, if present, is
    returned too (so --prompt can be omitted when the file is self-describing).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    prompt = None

    if torch.is_tensor(obj):
        latent = obj
    elif isinstance(obj, dict):
        prompt = obj.get("prompt")
        if "states" in obj:
            states = obj["states"]
            if states.ndim == 6:
                assert states.shape[0] == 1, f"expected leading batch dim 1, got {tuple(states.shape)}"
                states = states.squeeze(0)
            assert states.ndim == 5, f"unexpected 'states' shape {tuple(states.shape)} in {path}"
            latent = states[-1]  # clean endpoint, [F, C, H, W]
        else:
            latent = None
            for key in ("latent", "latents", "clean_latent", "video_latent"):
                if key in obj:
                    latent = obj[key]
                    break
            if latent is None:
                raise KeyError(
                    f"Could not find a latent tensor in {path}; expected one of "
                    f"'states'/'latent'/'latents'/'clean_latent'/'video_latent', got keys={list(obj.keys())}"
                )
    else:
        raise TypeError(f"Unsupported .pt content type in {path}: {type(obj)}")

    latent = latent.to(torch.float32)
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    assert latent.ndim == 5, f"expected latent shape [B, F, C, H, W] (or [F, C, H, W]), got {tuple(latent.shape)}"
    if latent.shape[0] != 1:
        print(f"[WARN] loaded latent has batch size {latent.shape[0]}; only index 0 will be used.")
        latent = latent[:1]

    got_fchw = tuple(latent.shape[1:])
    expected_fchw = tuple(latent_shape)
    assert got_fchw == expected_fchw, (
        f"latent shape {got_fchw} (F,C,H,W) does not match the pipeline's expected {expected_fchw}; "
        f"pass --latent_shape F C H W to override if this mismatch is intentional."
    )
    return latent, prompt


def save_video(video: torch.Tensor, output_path: str, fps: int):
    video = rearrange(video, "b t c h w -> b t h w c").cpu()
    video = (255.0 * video).clamp(0, 255).to(torch.uint8)
    write_video(output_path, video[0], fps=fps)
    print(f"[video] saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_latent_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                         help="Text conditioning for denoising. Optional if the .pt file has its own 'prompt' field.")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--label", type=str, default=None, help="Output filename prefix; defaults to the input file's stem.")
    parser.add_argument("--t_star", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                         help="Edit strengths as fractions of the full [0,1000] warped timestep schedule.")

    parser.add_argument("--latent_shape", type=int, nargs=4, default=[21, 16, 60, 104],
                         metavar=("F", "C", "H", "W"), help="Expected (and generator's native) latent shape.")
    parser.add_argument("--model_name", type=str, default="Wan2.1-T2V-1.3B")
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=48,
                         help="Steps in the NORMAL full schedule (t_star selects a tail of this, not a step count).")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional fine-tuned bidirectional generator checkpoint; omit to use raw pretrained Wan2.1 weights.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--save_input_reconstruction", action="store_true", default=True)
    parser.add_argument("--no_save_input_reconstruction", dest="save_input_reconstruction", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.input_latent_path).stem

    input_latent, embedded_prompt = load_input_latent(args.input_latent_path, latent_shape=args.latent_shape)
    prompt = args.prompt or embedded_prompt
    assert prompt, "No --prompt given and the .pt file has no embedded 'prompt' field."
    print(f"[input] {args.input_latent_path} shape={tuple(input_latent.shape)} prompt={prompt!r}")

    pipeline = SDEditStrengthPipeline(
        device=device,
        model_name=args.model_name,
        timestep_shift=args.timestep_shift,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latent_shape=tuple(args.latent_shape),
    )
    if args.checkpoint_path:
        pipeline.load_generator_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
    pipeline = pipeline.to(device=device, dtype=getattr(torch, args.dtype))
    pipeline.generator.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()

    input_latent = input_latent.to(device=device, dtype=pipeline.dtype)

    if args.save_input_reconstruction:
        input_video = pipeline.vae.decode_to_pixel(input_latent, use_cache=False)
        input_video = (input_video * 0.5 + 0.5).clamp(0, 1)
        pipeline.vae.model.clear_cache()
        save_video(input_video, str(output_folder / f"{label}_input_t0.00.mp4"), fps=args.fps)
    else:
        input_video = None

    for t_star in args.t_star:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

        video, _, info = pipeline.denoise_from_strength(
            input_latent, text_prompts=[prompt], t_star=t_star, generator=generator,
        )
        print(f"[t*={t_star}] start_timestep={info['start_timestep']:.1f} "
              f"steps={info['num_denoising_steps']}/{info['total_schedule_steps']}")

        if input_video is not None:
            pixel_diff = (video.cpu() - input_video.cpu()).abs().mean().item()
            print(f"[t*={t_star}] mean abs pixel difference vs. input reconstruction: {pixel_diff:.4f}")

        save_video(video, str(output_folder / f"{label}_t{t_star:.2f}.mp4"), fps=args.fps)

    print("[done]")


if __name__ == "__main__":
    main()
