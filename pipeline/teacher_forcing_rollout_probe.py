"""
Teacher-forced next-chunk probe vs. self-rollout probe for PrefixDMD (TeacherForcingDMD).

Context
-------
`model/teacher_forcing_dmd.py` (`TeacherForcingDMD.generator_loss` ->
`four_step_generate_with_grad`) is the method actually being optimized by
`trainer/tfdmd.py`. At every training step it:
  1. Prefills a read/write KV cache with a slice of GROUND TRUTH clean latents
     (`_prefill_context_cache`, timestep 0, `update_kv_cache=True`).
  2. Denoises ONE chunk of pure noise for `len(denoising_step_list)` steps,
     reading that cache but never writing to it (`update_kv_cache=False`), i.e.
     the chunk is generated purely conditioned on real history.
This script reuses exactly that calling convention (not the older
`pipeline/causal_inference.py` convention, which predates teacher forcing and
updates the cache differently) so the probe reflects what the generator was
actually taught to do.

Given an ODE trajectory .pt file (see `utils/dataset.py:ODERegressionPTDataset`,
`states` shaped [1, T_ode, 21, C, H, W] with the LAST entry along T_ode being
the clean endpoint), this script takes only that clean endpoint and runs two
diagnostics to separate two failure hypotheses for a checkpoint that degenerates
into flat color blocks:

  teacher_forced : every chunk is generated conditioned on REAL history only
                   (ground truth is always committed back to the cache,
                   regardless of what the model predicted). If this is already
                   broken, next-chunk prediction itself is broken -- the
                   problem is not rollout-specific.

  self_rollout   : standard autoregressive deployment. Each chunk is generated
                   conditioned on the model's OWN previous predictions (history
                   can optionally be seeded with a few leading ground-truth
                   chunks via --rollout_context_blocks). If this collapses
                   while teacher_forced stays healthy, the failure is exposure
                   bias / error accumulation in self-conditioned rollout.

  rollout_from_first_frame : minimal-grounding variant of self_rollout. Only
                   the single GT first frame is committed to the KV cache
                   (not a full num_frame_per_block chunk); every subsequent
                   chunk is generated and fed back self-conditioned. Separates
                   "needs a few real frames of anchoring" from "diverges
                   immediately even with one real anchor frame".

Usage
-----
python pipeline/teacher_forcing_rollout_probe.py \
    --config_path configs/prefix_dmd_framewise.yaml \
    --checkpoint_path training_outputs/.../checkpoint_model_000500/model.pt \
    --data_path dataset/vidprom_sample_16k/0.pt \
    --output_folder outputs/tf_rollout_probe/000500 \
    --mode both
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

try:
    from torchvision.io import write_video
except ImportError:
    # torchvision >=0.22 removed torchvision.io.write_video; fall back to imageio.
    import imageio

    def write_video(filename, video, fps):
        video = video.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
        imageio.mimwrite(filename, list(video), fps=fps, codec="libx264", quality=4.8)

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.misc import set_seed


class TFRolloutProbePipeline(torch.nn.Module):
    """
    Minimal causal-generation harness mirroring the exact KV-cache calling
    convention used at train time by `TeacherForcingDMD` (see module docstring),
    with two generation modes: always-ground-truth context ("teacher_forced")
    and always-self-generated context ("self_rollout").
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", True) else torch.float32

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        self.scheduler = self.generator.get_scheduler()

        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.generator.model.num_frame_per_block = self.num_frame_per_block
        assert not getattr(args, "independent_first_frame", False), \
            "independent_first_frame is not supported by this probe script"

        self.context_noise = getattr(args, "context_noise", 0)

        shape = getattr(args, "image_or_video_shape", [1, 21, 16, 60, 104])
        self.num_channels, self.height, self.width = shape[2], shape[3], shape[4]

    @property
    def frame_seq_length(self):
        patch_h, patch_w = self.generator.model.patch_size[1], self.generator.model.patch_size[2]
        return (self.height // patch_h) * (self.width // patch_w)

    def load_generator_checkpoint(self, checkpoint_path, use_ema=False):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = "generator_ema" if (use_ema and "generator_ema" in state_dict) else "generator"
        gen_sd = state_dict[key]
        try:
            self.generator.load_state_dict(gen_sd, strict=True)
        except RuntimeError:
            fixed = {}
            for k, v in gen_sd.items():
                if k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                fixed[k] = v
            self.generator.load_state_dict(fixed, strict=True)

    # ------------------------------------------------------------------
    # KV cache plumbing (mirrors model/teacher_forcing_dmd.py's
    # `_init_kv_cache` / `_initialize_crossattn_cache`)
    # ------------------------------------------------------------------
    def _init_kv_cache(self, batch_size, max_frames):
        model = self.generator.model
        num_layers = len(model.blocks)
        num_heads = model.num_heads
        head_dim = model.dim // model.num_heads
        kv_cache_size = max_frames * self.frame_seq_length

        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })
        return kv_cache

    def _init_crossattn_cache(self, batch_size):
        model = self.generator.model
        num_layers = len(model.blocks)
        num_heads = model.num_heads
        head_dim = model.dim // model.num_heads

        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "is_init": False,
            })
        return crossattn_cache

    @torch.no_grad()
    def _commit_chunk_to_cache(self, chunk, conditional_dict, kv_cache, crossattn_cache, current_start, context_timestep):
        """Write `chunk` into the KV cache as clean history at `current_start`."""
        batch_size, frame_len = chunk.shape[0], chunk.shape[1]
        timestep = torch.full([batch_size, frame_len], context_timestep, device=self.device, dtype=torch.long)
        self.generator(
            noisy_image_or_video=chunk,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            update_kv_cache=True,
        )

    @torch.no_grad()
    def _generate_chunk(self, conditional_dict, kv_cache, crossattn_cache, current_start, frame_len, batch_size):
        """
        Generate one chunk of `frame_len` frames by denoising pure noise with
        `self.denoising_step_list`, reading but never writing the KV cache --
        the exact read-only convention `four_step_generate_with_grad` uses.
        """
        noisy_input = torch.randn(
            [batch_size, frame_len, self.num_channels, self.height, self.width],
            device=self.device, dtype=self.dtype,
        )
        num_steps = len(self.denoising_step_list)
        x0 = None
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones([batch_size, frame_len], device=self.device, dtype=torch.int64) * current_timestep

            _, x0 = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                update_kv_cache=False,
            )

            if index < num_steps - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    x0.flatten(0, 1),
                    torch.randn_like(x0.flatten(0, 1)),
                    torch.full(
                        [batch_size * frame_len], next_timestep, device=self.device, dtype=torch.float32,
                    ),
                ).unflatten(0, x0.shape[:2])
        return x0

    # ------------------------------------------------------------------
    # Probe 1: teacher-forced next-chunk prediction (no self-rollout at all)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_teacher_forced(self, clean_latent, conditional_dict, num_blocks=None):
        """
        clean_latent: [1, T, C, H, W] ground-truth latents.
        At every position the KV cache holds only REAL history (ground truth
        up to that point); the model's own prediction is recorded but never
        fed back. Isolates "is next-chunk prediction itself broken".
        """
        batch_size, total_frames = clean_latent.shape[0], clean_latent.shape[1]
        block = self.num_frame_per_block
        num_blocks = num_blocks or (total_frames // block)
        assert total_frames % block == 0, f"{total_frames} frames not divisible by block size {block}"
        assert num_blocks * block <= total_frames

        kv_cache = self._init_kv_cache(batch_size, max_frames=num_blocks * block)
        crossattn_cache = self._init_crossattn_cache(batch_size)

        predicted = torch.zeros_like(clean_latent[:, :num_blocks * block])
        per_chunk_mse = []
        current_start = 0
        for block_idx in range(num_blocks):
            frame_slice = slice(block_idx * block, block_idx * block + block)

            pred_chunk = self._generate_chunk(
                conditional_dict, kv_cache, crossattn_cache, current_start, block, batch_size,
            )
            predicted[:, frame_slice] = pred_chunk

            gt_chunk = clean_latent[:, frame_slice]
            per_chunk_mse.append(
                torch.nn.functional.mse_loss(pred_chunk.float(), gt_chunk.float()).item()
            )

            # Teacher forcing: commit the REAL frame, never the model's own prediction.
            self._commit_chunk_to_cache(
                gt_chunk, conditional_dict, kv_cache, crossattn_cache, current_start, context_timestep=0,
            )
            current_start += block * self.frame_seq_length

        return predicted, per_chunk_mse

    # ------------------------------------------------------------------
    # Probe 2: standard autoregressive self-rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_self_rollout(self, clean_latent, conditional_dict, num_blocks, context_blocks=0):
        """
        clean_latent: [1, T, C, H, W] ground-truth latents, used only to seed
        the first `context_blocks` chunks (0 = pure from-scratch rollout,
        matching normal T2V deployment). Every later chunk is generated from,
        and fed back as, the model's OWN previous prediction. Isolates "does
        error accumulate once the model must trust itself".
        """
        batch_size = clean_latent.shape[0]
        block = self.num_frame_per_block
        assert context_blocks <= num_blocks
        assert context_blocks * block <= clean_latent.shape[1], \
            "not enough ground-truth frames available to seed the requested context_blocks"

        kv_cache = self._init_kv_cache(batch_size, max_frames=num_blocks * block)
        crossattn_cache = self._init_crossattn_cache(batch_size)

        output = torch.zeros(
            [batch_size, num_blocks * block, self.num_channels, self.height, self.width],
            device=self.device, dtype=self.dtype,
        )

        current_start = 0
        for block_idx in range(num_blocks):
            frame_slice = slice(block_idx * block, block_idx * block + block)

            if block_idx < context_blocks:
                chunk = clean_latent[:, frame_slice]
                output[:, frame_slice] = chunk
                self._commit_chunk_to_cache(
                    chunk, conditional_dict, kv_cache, crossattn_cache, current_start, context_timestep=0,
                )
            else:
                chunk = self._generate_chunk(
                    conditional_dict, kv_cache, crossattn_cache, current_start, block, batch_size,
                )
                output[:, frame_slice] = chunk
                # Self-rollout: commit the model's OWN prediction, not ground truth.
                self._commit_chunk_to_cache(
                    chunk, conditional_dict, kv_cache, crossattn_cache, current_start,
                    context_timestep=self.context_noise,
                )
            current_start += block * self.frame_seq_length

        return output

    # ------------------------------------------------------------------
    # Probe 3: self-rollout grounded in only the single GT first frame
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_rollout_from_first_frame(self, clean_latent, conditional_dict, num_blocks):
        """
        Seed the KV cache with only the single ground-truth FIRST FRAME (not a
        full `num_frame_per_block` chunk, unlike `run_self_rollout`'s
        `context_blocks`), then roll out every subsequent chunk
        autoregressively from the model's OWN predictions. This is the
        minimal-grounding rollout probe: if `self_rollout` seeded with a whole
        GT chunk stays healthy but this collapses, the model needs more than
        one real frame of anchoring before self-conditioning becomes unstable.
        """
        batch_size = clean_latent.shape[0]
        block = self.num_frame_per_block

        kv_cache = self._init_kv_cache(batch_size, max_frames=1 + num_blocks * block)
        crossattn_cache = self._init_crossattn_cache(batch_size)

        output = torch.zeros(
            [batch_size, 1 + num_blocks * block, self.num_channels, self.height, self.width],
            device=self.device, dtype=self.dtype,
        )

        first_frame = clean_latent[:, :1]
        output[:, :1] = first_frame
        self._commit_chunk_to_cache(
            first_frame, conditional_dict, kv_cache, crossattn_cache, current_start=0, context_timestep=0,
        )
        current_start = self.frame_seq_length

        for block_idx in range(num_blocks):
            frame_slice = slice(1 + block_idx * block, 1 + block_idx * block + block)

            chunk = self._generate_chunk(
                conditional_dict, kv_cache, crossattn_cache, current_start, block, batch_size,
            )
            output[:, frame_slice] = chunk
            # Self-rollout: commit the model's OWN prediction, not ground truth.
            self._commit_chunk_to_cache(
                chunk, conditional_dict, kv_cache, crossattn_cache, current_start,
                context_timestep=self.context_noise,
            )
            current_start += block * self.frame_seq_length

        return output

    @torch.no_grad()
    def decode(self, latent):
        video = self.vae.decode_to_pixel(latent.to(self.dtype), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()
        return video


def load_pt_sample(data_path, sample_idx=0):
    """
    Load one ODE-trajectory sample saved as described in
    `utils/dataset.py:ODERegressionPTDataset`, and return only its clean
    endpoint latent, as [1, T, C, H, W].
    """
    path = Path(data_path)
    if path.is_file():
        file_path = path
    else:
        def sort_key(p):
            return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)
        files = sorted(path.glob("*.pt"), key=sort_key)
        if not files:
            raise FileNotFoundError(f"No .pt files found under {data_path}")
        file_path = files[sample_idx]

    sample = torch.load(file_path, map_location="cpu", weights_only=False)
    states = sample["states"]
    if states.ndim == 6:
        assert states.shape[0] == 1, f"expected leading batch dim 1, got {tuple(states.shape)}"
        states = states.squeeze(0)
    assert states.ndim == 5, f"unexpected states shape {tuple(states.shape)} in {file_path}"

    clean_latent = states[-1].unsqueeze(0).to(torch.float32)  # [1, F, C, H, W], the final (clean) ODE step
    prompt = sample["prompt"]
    print(f"[data] Loaded {file_path} (prompt={prompt!r}, trajectory shape={tuple(states.shape)})")
    return clean_latent, prompt


def save_video(video, output_path, fps=16):
    video = rearrange(video, "b t c h w -> b t h w c").cpu()
    video = (255.0 * video).clamp(0, 255).to(torch.uint8)
    write_video(str(output_path), video[0], fps=fps)
    print(f"[video] saved -> {output_path}")


def copy_shortcut_report(tag, generated, gt_full, start_offset, block, boundary_offset=0):
    """
    "Copy-last-frame shortcut" diagnostic.

    For every generated frame at global timeline position g = start_offset + i
    (skipping g == 0, which has no previous frame, and any g the ground-truth
    trajectory does not cover), computes three quantities:
        copy_baseline = MSE(gt[g-1], gt[g])      -- cost of trivially copying the previous GT frame
        MSE(pred, gt)                            -- actual prediction error
        MSE(pred, gt[g-1])                       -- how close the prediction is to the previous frame instead

    If MSE(pred, gt) tracks copy_baseline, and MSE(pred, gt[g-1]) < MSE(pred, gt)
    (pred lands closer to the previous frame than to the true next frame), the
    model is taking a "copy the last frame" shortcut rather than predicting
    real motion. Frames are also split into "boundary" (first frame of a
    num_frame_per_block leaf, i.e. a cross-chunk step) vs. "interior" (motion
    predicted entirely within one leaf) to check for a "moves within a chunk,
    freezes across chunk boundaries" pattern.

    Args:
        generated: [1, N, C, H, W] predicted frames occupying global
            timeline positions [start_offset, start_offset + N). Pass only
            the model-generated slice (exclude any ground-truth-seeded
            context frames, which would trivially report zero error).
        gt_full: [1, T, C, H, W] ground-truth latents on the same timeline.
            T may be shorter than start_offset + N (e.g. a long-horizon
            rollout stress test); uncovered frames are silently skipped.
        start_offset: index into gt_full where generated[:, 0] lives.
        block: num_frame_per_block, used for the boundary/interior split.
        boundary_offset: global index of the first leaf boundary (0 unless
            a single ungrouped frame, e.g. one GT anchor frame, precedes the
            first leaf -- then leaves start at 1, 1+block, 1+2*block, ...).
    """
    rows = []
    for i in range(generated.shape[1]):
        g = start_offset + i
        if g == 0 or g >= gt_full.shape[1]:
            continue
        gt_cur = gt_full[:, g].float()
        gt_prev = gt_full[:, g - 1].float()
        pred_cur = generated[:, i].float()
        rows.append((
            g,
            (g - boundary_offset) % block == 0,  # first frame of a leaf -> crosses a chunk boundary
            F.mse_loss(gt_prev, gt_cur).item(),
            F.mse_loss(pred_cur, gt_cur).item(),
            F.mse_loss(pred_cur, gt_prev).item(),
        ))

    print(f"\n[{tag}] copy-shortcut diagnostic")
    if not rows:
        print("  (no overlapping prediction/ground-truth frames available)")
        return

    def summarize(subset, label):
        if not subset:
            print(f"  [{label:8s}] n=0")
            return
        n = len(subset)
        avg_copy = sum(r[2] for r in subset) / n
        avg_pred_gt = sum(r[3] for r in subset) / n
        avg_pred_prev = sum(r[4] for r in subset) / n
        closer_to_prev = sum(1 for r in subset if r[4] < r[3])
        print(
            f"  [{label:8s}] n={n:3d}  copy_baseline={avg_copy:.6f}  "
            f"MSE(pred,gt)={avg_pred_gt:.6f}  MSE(pred,prev)={avg_pred_prev:.6f}  "
            f"closer_to_prev={closer_to_prev}/{n}"
        )

    summarize(rows, "all")
    summarize([r for r in rows if r[1]], "boundary")
    summarize([r for r in rows if not r[1]], "interior")

    print("  per-frame: [B]=chunk-boundary frame   copy_baseline | MSE(pred,gt) | MSE(pred,prev)")
    for g, is_boundary, copy_baseline, mse_pred_gt, mse_pred_prev in rows:
        marker = "B" if is_boundary else " "
        flag = "  <- closer to prev frame than to gt" if mse_pred_prev < mse_pred_gt else ""
        print(
            f"    frame {g:03d} [{marker}]  {copy_baseline:.6f} | {mse_pred_gt:.6f} | {mse_pred_prev:.6f}{flag}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="A single ODE .pt sample, or a directory of them")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which .pt file to use when --data_path is a directory")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["teacher_forced", "self_rollout", "rollout_from_first_frame", "both", "all"],
        default="both",
        help="'both' runs teacher_forced + self_rollout; 'all' additionally runs rollout_from_first_frame.",
    )
    parser.add_argument(
        "--num_rollout_blocks", type=int, default=None,
        help="Number of chunks to roll out in self_rollout / rollout_from_first_frame mode. Default: as many "
             "chunks as available in the GT trajectory. Set higher to stress-test longer-horizon collapse.",
    )
    parser.add_argument(
        "--rollout_context_blocks", type=int, default=0,
        help="Number of leading ground-truth chunks used to seed self_rollout (0 = pure from-scratch rollout).",
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    default_config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml"
    config = OmegaConf.merge(OmegaConf.load(default_config_path), OmegaConf.load(args.config_path))

    clean_latent, prompt = load_pt_sample(args.data_path, args.sample_idx)
    clean_latent = clean_latent.to(device=device)

    pipeline = TFRolloutProbePipeline(config, device=device)
    pipeline.load_generator_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
    pipeline = pipeline.to(device=device, dtype=pipeline.dtype)
    pipeline.generator.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()

    clean_latent = clean_latent.to(dtype=pipeline.dtype)

    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    num_blocks_in_gt = clean_latent.shape[1] // pipeline.num_frame_per_block

    # Always render the ground-truth reconstruction once, for visual reference.
    gt_video = pipeline.decode(clean_latent)
    save_video(gt_video, output_folder / "ground_truth.mp4", fps=args.fps)

    if args.mode in ("teacher_forced", "both"):
        predicted, per_chunk_mse = pipeline.run_teacher_forced(
            clean_latent, conditional_dict, num_blocks=num_blocks_in_gt,
        )
        print("[teacher_forced] per-chunk MSE against ground truth (always-real context):")
        for i, mse in enumerate(per_chunk_mse):
            print(f"  chunk {i:02d}: {mse:.6f}")
        copy_shortcut_report(
            "teacher_forced", predicted, clean_latent, start_offset=0, block=pipeline.num_frame_per_block,
        )
        tf_video = pipeline.decode(predicted)
        save_video(tf_video, output_folder / "teacher_forced.mp4", fps=args.fps)

    if args.mode in ("self_rollout", "both", "all"):
        num_rollout_blocks = args.num_rollout_blocks or num_blocks_in_gt
        context_blocks = args.rollout_context_blocks
        block = pipeline.num_frame_per_block
        rollout = pipeline.run_self_rollout(
            clean_latent, conditional_dict,
            num_blocks=num_rollout_blocks,
            context_blocks=context_blocks,
        )
        copy_shortcut_report(
            "self_rollout", rollout[:, context_blocks * block:], clean_latent,
            start_offset=context_blocks * block, block=block,
        )
        rollout_video = pipeline.decode(rollout)
        save_video(rollout_video, output_folder / "self_rollout.mp4", fps=args.fps)

    if args.mode in ("rollout_from_first_frame", "all"):
        num_rollout_blocks = args.num_rollout_blocks or num_blocks_in_gt
        block = pipeline.num_frame_per_block
        first_frame_rollout = pipeline.run_rollout_from_first_frame(
            clean_latent, conditional_dict, num_blocks=num_rollout_blocks,
        )
        copy_shortcut_report(
            "rollout_from_first_frame", first_frame_rollout[:, 1:], clean_latent,
            start_offset=1, block=block, boundary_offset=1,
        )
        first_frame_video = pipeline.decode(first_frame_rollout)
        save_video(first_frame_video, output_folder / "rollout_from_first_frame.mp4", fps=args.fps)

    print("[done]")


if __name__ == "__main__":
    main()
