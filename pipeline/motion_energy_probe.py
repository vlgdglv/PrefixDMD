"""
D1 probe: chunk-wise motion-energy comparison across three groups, to localize
whether a causal-model init method's dynamic-degree collapse is a "startup
stall" (chunk 0 specifically under-moves, later chunks continue at chunk 0's
own level) or something deeper (motion stays dead / never recovers, or is
already dead in chunk 0 of the healthy baselines too -- which would point
elsewhere).

Groups (run this script once per group with --source, each producing a
`motion_energy_<label>.json` report in --output_folder; then run once more
with --source compare to merge them into one side-by-side table):

  causal : your TF-DMD / PrefixDMD causal generator (`TeacherForcingDMD`,
           see `pipeline/teacher_forcing_rollout_probe.py`), rolled out from
           scratch (no GT context) for --num_videos prompts.
  cf     : the "causal forcing" baseline this repo was forked from
           (`pipeline/causal_inference.py:CausalInferencePipeline`), same
           from-scratch rollout, for a controlled comparison.
  gt     : ground-truth clean latents from the 16k ODE training set
           (`utils/dataset.py:ODERegressionPTDataset`'s .pt format),
           decoded directly -- no model involved. Establishes what "healthy"
           motion looks like at this exact chunk layout.
  compare: merge previously-saved reports into one table. Pure stdlib + json
           -- unlike the other three sources, this needs none of the
           torch/einops/etc. ML stack, so it also works on a plain laptop.

For a fair, paired comparison, causal/cf/gt all draw from the SAME
--data_path .pt files at the same --sample_start_idx/--num_videos: gt decodes
their stored clean latents directly, causal/cf reuse their stored prompts as
generation conditioning.

Motion energy = RAFT optical-flow magnitude between consecutive PIXEL frames
(same methodology as VBench's dynamic_degree), kept as a continuous
per-frame-pair curve (not thresholded into a dynamic/static binary -- see
`utils/motion_energy.py` for why), then split into "interior" (motion within
one num_frame_per_block leaf) vs "boundary" (the transition crossing into
the next leaf) frame-pairs and averaged across videos, per chunk.

Usage:
    # 1) your causal model
    python pipeline/motion_energy_probe.py --source causal \
        --config_path configs/prefix_dmd_chunkwise.yaml \
        --checkpoint_path training_outputs/.../checkpoint_model_002400/model.pt \
        --data_path dataset/vidprom_sample_16k --num_videos 50 \
        --output_folder outputs/motion_energy --label causal_init_2400

    # 2) CF baseline
    python pipeline/motion_energy_probe.py --source cf \
        --config_path configs/causal_forcing_dmd_chunkwise.yaml \
        --checkpoint_path checkpoints/chunkwise/causal_ode.pt \
        --data_path dataset/vidprom_sample_16k --num_videos 50 \
        --output_folder outputs/motion_energy --label cf_init

    # 3) ground truth (block must match whichever generator you're diagnosing)
    python pipeline/motion_energy_probe.py --source gt --block 3 \
        --data_path dataset/vidprom_sample_16k --num_videos 50 \
        --output_folder outputs/motion_energy --label gt_16k

    # 4) merge
    python pipeline/motion_energy_probe.py --source compare \
        --reports outputs/motion_energy/motion_energy_causal_init_2400.json \
                  outputs/motion_energy/motion_energy_cf_init.json \
                  outputs/motion_energy/motion_energy_gt_16k.json
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _decode_video(vae, latent, dtype):
    from einops import rearrange
    video = vae.decode_to_pixel(latent.to(dtype), use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    vae.model.clear_cache()
    return rearrange(video, "b t c h w -> b t h w c")[0].float().cpu()


def load_prompts(args, indices):
    from pipeline.teacher_forcing_rollout_probe import load_pt_sample
    prompts = []
    for idx in indices:
        _, prompt = load_pt_sample(args.data_path, sample_idx=idx)
        prompts.append(prompt)
    return prompts


def generate_causal_videos(args, device, prompts):
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from utils.misc import set_seed
    from pipeline.teacher_forcing_rollout_probe import TFRolloutProbePipeline

    config = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs" / "default_config.yaml"),
        OmegaConf.load(args.config_path),
    )

    pipeline = TFRolloutProbePipeline(config, device=device)
    pipeline.load_generator_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
    pipeline = pipeline.to(device=device, dtype=pipeline.dtype)
    pipeline.generator.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()

    block = pipeline.num_frame_per_block
    num_latent_frames = getattr(config, "num_training_frames", 21)
    num_blocks = num_latent_frames // block
    dummy_clean = torch.zeros(
        [1, num_blocks * block, pipeline.num_channels, pipeline.height, pipeline.width],
        device=device, dtype=pipeline.dtype,
    )

    videos = []
    for i, prompt in enumerate(tqdm(prompts, desc="[causal] generating")):
        set_seed(args.seed + i)
        conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
        rollout = pipeline.run_self_rollout(dummy_clean, conditional_dict, num_blocks=num_blocks, context_blocks=0)
        videos.append(_decode_video(pipeline.vae, rollout, pipeline.dtype))
    return videos, block, num_latent_frames


def generate_cf_videos(args, device, prompts):
    import torch
    from einops import rearrange
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from utils.misc import set_seed
    from pipeline.causal_inference import CausalInferencePipeline

    config = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs" / "default_config.yaml"),
        OmegaConf.load(args.config_path),
    )

    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if (args.use_ema and "generator_ema" in state_dict) else "generator"
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
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    block = pipeline.num_frame_per_block
    num_latent_frames = list(getattr(config, "image_or_video_shape", [1, 21, 16, 60, 104]))[1]

    videos = []
    for i, prompt in enumerate(tqdm(prompts, desc="[cf] generating")):
        set_seed(args.seed + i)
        noise = torch.randn([1, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
        video = pipeline.inference(noise=noise, text_prompts=[prompt], initial_latent=None)
        pipeline.vae.model.clear_cache()
        videos.append(rearrange(video, "b t c h w -> b t h w c")[0].float().cpu())
    return videos, block, num_latent_frames


def load_gt_videos(args, device, indices):
    import torch
    from tqdm import tqdm
    from pipeline.teacher_forcing_rollout_probe import load_pt_sample
    from utils.wan_wrapper import WanVAEWrapper

    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    videos, prompts, num_latent_frames = [], [], None
    for idx in tqdm(indices, desc="[gt] loading"):
        clean_latent, prompt = load_pt_sample(args.data_path, sample_idx=idx)
        clean_latent = clean_latent.to(device=device)
        num_latent_frames = clean_latent.shape[1]
        videos.append(_decode_video(vae, clean_latent, torch.bfloat16))
        prompts.append(prompt)
    return videos, prompts, num_latent_frames


def run_generate(args):
    import os

    import torch
    from tqdm import tqdm
    from utils.misc import set_seed
    from utils.motion_energy import (
        aggregate_across_videos,
        chunk_rows_for_video,
        compute_pairwise_flow_magnitude,
        compute_pairwise_flow_magnitude_farneback,
        load_flow_model,
        num_pixel_frames,
        print_chunk_summary,
    )
    try:
        from torchvision.io import write_video
    except ImportError:
        import imageio

        def write_video(filename, video, fps):
            video = video.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
            quality = float(os.environ.get("VIDEO_QUALITY", "4.8"))
            imageio.mimwrite(filename, list(video), fps=fps, codec="libx264", quality=quality)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    label = args.label or args.source

    indices = list(range(args.sample_start_idx, args.sample_start_idx + args.num_videos))

    if args.source == "gt":
        assert args.block is not None, "--block is required for --source gt (pass the same block size as the generator you're comparing against)"
        videos, prompts, num_latent_frames = load_gt_videos(args, device, indices)
        block = args.block
    else:
        assert args.config_path and args.checkpoint_path, "--config_path and --checkpoint_path are required for --source causal/cf"
        prompts = load_prompts(args, indices)
        if args.source == "causal":
            videos, block, num_latent_frames = generate_causal_videos(args, device, prompts)
        else:
            videos, block, num_latent_frames = generate_cf_videos(args, device, prompts)

    print(f"[{label}] {len(videos)} videos, num_frame_per_block={block}, num_latent_frames={num_latent_frames}, "
          f"pixel_frames={num_pixel_frames(num_latent_frames)}")

    if args.flow_method == "farneback":
        flow_model = None
    else:
        try:
            flow_model = load_flow_model(device, model_name=args.flow_method)
        except Exception as e:
            print(f"[WARN] failed to load {args.flow_method} ({e}); falling back to Farneback optical flow.")
            flow_model = None
            args.flow_method = "farneback"

    resize_hw = (args.flow_resize_height, args.flow_resize_width)

    per_video_rows, per_video_curves = [], []
    for video in tqdm(videos, desc=f"[{label}] computing motion energy"):
        if flow_model is not None:
            curve = compute_pairwise_flow_magnitude(
                video, flow_model, device, resize_hw=resize_hw, batch_size=args.flow_batch_size,
            )
        else:
            curve = compute_pairwise_flow_magnitude_farneback(video, resize_hw=resize_hw)
        curve = curve.tolist()
        per_video_curves.append(curve)
        per_video_rows.append(chunk_rows_for_video(curve, block=block, num_latent_frames=num_latent_frames))

    summary = aggregate_across_videos(per_video_rows)
    print_chunk_summary(label, summary)

    overall_interior = [r["interior_mean"] for r in summary if r["interior_mean"] is not None]
    overall_boundary = [r["boundary_mean"] for r in summary if r["boundary_mean"] is not None]

    report = {
        "label": label,
        "source": args.source,
        "num_videos": len(videos),
        "block": block,
        "num_latent_frames": num_latent_frames,
        "flow_method": args.flow_method,
        "flow_resize_hw": list(resize_hw),
        "prompts": prompts,
        "per_chunk_summary": summary,
        "per_video_curves": per_video_curves,
        "overall_interior_mean": (sum(overall_interior) / len(overall_interior)) if overall_interior else None,
        "overall_boundary_mean": (sum(overall_boundary) / len(overall_boundary)) if overall_boundary else None,
    }
    report_path = output_folder / f"motion_energy_{label}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[report] saved -> {report_path}")

    for i in range(min(args.save_sample_videos, len(videos))):
        video_uint8 = (videos[i] * 255.0).clamp(0, 255).to(torch.uint8)
        out_path = output_folder / f"{label}_sample_{i:02d}.mp4"
        write_video(str(out_path), video_uint8, fps=args.fps)
        print(f"[video] saved -> {out_path}")


def run_compare(args):
    """Pure stdlib + json -- no torch/einops/etc. needed, so this also runs on a plain laptop."""
    assert args.reports, "--reports is required for --source compare"
    reports = []
    for path in args.reports:
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    labels = args.labels or [r["label"] for r in reports]
    assert len(labels) == len(reports), "--labels must have the same length as --reports"

    blocks = {r["block"] for r in reports}
    num_latents = {r["num_latent_frames"] for r in reports}
    if len(blocks) > 1 or len(num_latents) > 1:
        print(f"[WARN] reports don't share the same block/num_latent_frames ({blocks}, {num_latents}) "
              f"-- chunk indices may not be directly comparable.")

    num_chunks = max(len(r["per_chunk_summary"]) for r in reports)
    print(f"\n[compare] {', '.join(labels)}")
    col_width = 22
    header = f"  {'chunk':>5}"
    for label in labels:
        header += f" | {(label + ' interior'):>{col_width}} {(label + ' boundary'):>{col_width}}"
    print(header)
    for c in range(num_chunks):
        row = f"  {c:>5}"
        for r in reports:
            if c < len(r["per_chunk_summary"]):
                cell = r["per_chunk_summary"][c]
                im = f"{cell['interior_mean']:.4f}" if cell["interior_mean"] is not None else "n/a"
                bm = f"{cell['boundary_mean']:.4f}" if cell["boundary_mean"] is not None else "n/a"
            else:
                im, bm = "n/a", "n/a"
            row += f" | {im:>{col_width}} {bm:>{col_width}}"
        print(row)

    print()
    for label, r in zip(labels, reports):
        oi = r["overall_interior_mean"]
        ob = r["overall_boundary_mean"]
        print(f"  [{label}] overall_interior_mean={oi if oi is None else round(oi, 4)}  "
              f"overall_boundary_mean={ob if ob is None else round(ob, 4)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["causal", "cf", "gt", "compare"], required=True)
    parser.add_argument("--data_path", type=str, default="dataset/vidprom_sample_16k",
                         help="Directory of ODE .pt samples: the GT video source, and (for causal/cf) the "
                              "paired prompt source so all groups are driven by the same prompts.")
    parser.add_argument("--sample_start_idx", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=16)
    parser.add_argument("--output_folder", type=str, help="Required for causal/cf/gt.")
    parser.add_argument("--label", type=str, default=None, help="Report name; defaults to --source.")

    parser.add_argument("--config_path", type=str, help="Required for --source causal/cf.")
    parser.add_argument("--checkpoint_path", type=str, help="Required for --source causal/cf.")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--block", type=int, default=None,
        help="num_frame_per_block for chunk aggregation. Required for --source gt (ground truth has no "
             "inherent chunking -- pass the same block size as whichever causal/cf run you're comparing "
             "against). Auto-read from the pipeline config for --source causal/cf.",
    )

    parser.add_argument("--flow_method", choices=["raft_small", "raft_large", "farneback"], default="raft_small")
    parser.add_argument("--flow_resize_height", type=int, default=240)
    parser.add_argument("--flow_resize_width", type=int, default=416)
    parser.add_argument("--flow_batch_size", type=int, default=16)
    parser.add_argument("--save_sample_videos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)

    parser.add_argument("--reports", type=str, nargs="+", help="--source compare: JSON reports to merge.")
    parser.add_argument("--labels", type=str, nargs="+", help="--source compare: labels for --reports (default: each report's own saved label).")

    args = parser.parse_args()

    if args.source == "compare":
        run_compare(args)
    else:
        assert args.output_folder, "--output_folder is required for --source causal/cf/gt"
        run_generate(args)


if __name__ == "__main__":
    main()
