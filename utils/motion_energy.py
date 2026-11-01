"""
Motion-energy measurement for the D1 "chunk-wise dynamic-degree" probe.

Methodology note (why this isn't literally VBench's dynamic_degree metric):
VBench's dynamic_degree estimates per-frame-pair displacement with RAFT optical
flow, then thresholds each video into a binary dynamic/static label and
reports the dynamic fraction across a dataset -- a coarse, dataset-level
summary. D1 needs to localize *where in the chunk structure* motion collapses
(does chunk 0 start low and do later chunks recover, or is it uniformly
dead?), which that binarization would destroy. So this module keeps the same
RAFT-based measurement VBench uses, but reports the raw per-frame-pair flow
magnitude as a continuous curve, then splits that curve into "interior"
(motion predicted within one `num_frame_per_block` leaf) vs "boundary"
(the single transition crossing into the next leaf) frame-pairs, mirroring
the same boundary/interior split used by `copy_shortcut_report` in
`pipeline/teacher_forcing_rollout_probe.py`, just on pixel-space optical flow
instead of latent MSE.

Latent <-> pixel frame mapping: the Wan2.1 VAE maps `F` latent frames to
`1 + 4*(F-1)` pixel frames -- the first latent frame maps to exactly 1 pixel
frame (treated like an image), and every later latent frame maps to 4
consecutive pixel frames. Chunk boundaries are defined on LATENT frame
indices (every `num_frame_per_block` latent frames), so they must be mapped
through this scheme to the pixel-frame ranges the flow curve is indexed by.
"""
import statistics
from typing import List, Optional, Sequence

import torch


# ----------------------------------------------------------------------
# Latent <-> pixel frame mapping (Wan2.1 VAE: F latents -> 1 + 4*(F-1) pixels)
# ----------------------------------------------------------------------
def pixel_range_for_latent_index(latent_index: int):
    """[start, end) pixel-frame range contributed by a single latent frame."""
    if latent_index == 0:
        return (0, 1)
    start = 1 + 4 * (latent_index - 1)
    return (start, start + 4)


def pixel_range_for_chunk(chunk_start_latent: int, block: int):
    """[start, end) pixel-frame range spanned by `block` latent frames starting at `chunk_start_latent`."""
    first_start, _ = pixel_range_for_latent_index(chunk_start_latent)
    _, last_end = pixel_range_for_latent_index(chunk_start_latent + block - 1)
    return first_start, last_end


def num_pixel_frames(num_latent_frames: int) -> int:
    return 1 + 4 * (num_latent_frames - 1)


# ----------------------------------------------------------------------
# Optical flow backends
# ----------------------------------------------------------------------
def load_flow_model(device, model_name: str = "raft_small"):
    """
    model_name: "raft_small" or "raft_large" (torchvision pretrained RAFT,
    the same flow methodology VBench's dynamic_degree uses). Requires network
    access on first call to download pretrained weights; use
    `compute_pairwise_flow_magnitude_farneback` instead if that's unavailable.
    """
    from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights

    if model_name == "raft_small":
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
    elif model_name == "raft_large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    else:
        raise ValueError(f"Unknown flow model: {model_name}")
    return model.to(device).eval()


@torch.no_grad()
def compute_pairwise_flow_magnitude(
    frames: torch.Tensor,
    flow_model,
    device,
    resize_hw: Optional[Sequence[int]] = (240, 416),
    batch_size: int = 16,
) -> torch.Tensor:
    """
    frames: [T, H, W, 3] float tensor in [0, 1] (decoded pixel video, single sample).
    Returns: [T-1] tensor, mean spatial flow magnitude per consecutive frame
    pair (in resized-pixel units -- consistent across groups as long as the
    same `resize_hw` is used for all of them, which the probe script enforces).
    """
    assert frames.shape[0] >= 2, "need at least 2 pixel frames to compute motion"
    frames = frames.permute(0, 3, 1, 2).to(device).float()  # [T, 3, H, W]
    if resize_hw is not None:
        assert resize_hw[0] % 8 == 0 and resize_hw[1] % 8 == 0, "RAFT requires dims divisible by 8"
        frames = torch.nn.functional.interpolate(frames, size=tuple(resize_hw), mode="bilinear", align_corners=False)
    frames = frames * 2.0 - 1.0  # [0, 1] -> [-1, 1], standard RAFT input range

    img1, img2 = frames[:-1], frames[1:]
    mags = []
    for start in range(0, img1.shape[0], batch_size):
        b1, b2 = img1[start:start + batch_size], img2[start:start + batch_size]
        flow = flow_model(b1, b2)[-1]  # last refinement iteration, [b, 2, H, W]
        mag = flow.pow(2).sum(dim=1).sqrt()  # [b, H, W]
        mags.append(mag.mean(dim=(1, 2)).cpu())
    return torch.cat(mags, dim=0)  # [T-1]


def compute_pairwise_flow_magnitude_farneback(
    frames: torch.Tensor,
    resize_hw: Optional[Sequence[int]] = (240, 416),
) -> torch.Tensor:
    """
    OpenCV Farneback fallback -- no pretrained-weight download required, for
    offline clusters. Less accurate than RAFT but same "continuous flow
    magnitude curve" semantics, so group-to-group comparisons remain valid.
    """
    import cv2
    import numpy as np

    assert frames.shape[0] >= 2
    frames_np = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [T, H, W, 3]

    mags = []
    prev_gray = None
    for t in range(frames_np.shape[0]):
        frame = frames_np[t]
        if resize_hw is not None:
            frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            mags.append(mag)
        prev_gray = gray
    return torch.tensor(mags, dtype=torch.float32)


# ----------------------------------------------------------------------
# Chunk-wise aggregation
# ----------------------------------------------------------------------
def chunk_rows_for_video(flow_curve: Sequence[float], block: int, num_latent_frames: int) -> List[dict]:
    """
    Split one video's [T_pixel - 1] flow curve into per-chunk rows.

    Each row: {chunk, pixel_range, interior_mean (None if the chunk has <2
    pixel frames, e.g. chunk 0 under framewise block=1), interior_n,
    boundary (None for chunk 0, which has no incoming transition)}.
    """
    assert num_latent_frames % block == 0
    num_chunks = num_latent_frames // block
    rows = []
    for c in range(num_chunks):
        chunk_start_latent = c * block
        px_start, px_end = pixel_range_for_chunk(chunk_start_latent, block)
        interior_idx = [i for i in range(px_start, px_end - 1) if i < len(flow_curve)]
        interior_vals = [float(flow_curve[i]) for i in interior_idx]
        boundary_idx = px_start - 1
        boundary_val = float(flow_curve[boundary_idx]) if 0 <= boundary_idx < len(flow_curve) else None
        rows.append({
            "chunk": c,
            "pixel_range": [px_start, px_end],
            "interior_mean": (sum(interior_vals) / len(interior_vals)) if interior_vals else None,
            "interior_n": len(interior_vals),
            "boundary": boundary_val,
        })
    return rows


def _mean_std(values: List[float]):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return values[0], 0.0, 1
    return statistics.mean(values), statistics.stdev(values), len(values)


def aggregate_across_videos(per_video_rows: List[List[dict]]) -> List[dict]:
    """
    per_video_rows: one `chunk_rows_for_video(...)` output per video (same
    block/num_latent_frames, so same chunk count/layout).
    Returns per-chunk {chunk, pixel_range, interior_mean, interior_std,
    interior_n_videos, boundary_mean, boundary_std, boundary_n_videos}.
    """
    num_chunks = len(per_video_rows[0])
    summary = []
    for c in range(num_chunks):
        interior_vals = [row[c]["interior_mean"] for row in per_video_rows]
        boundary_vals = [row[c]["boundary"] for row in per_video_rows]
        im, istd, in_ = _mean_std(interior_vals)
        bm, bstd, bn = _mean_std(boundary_vals)
        summary.append({
            "chunk": c,
            "pixel_range": per_video_rows[0][c]["pixel_range"],
            "interior_mean": im, "interior_std": istd, "interior_n_videos": in_,
            "boundary_mean": bm, "boundary_std": bstd, "boundary_n_videos": bn,
        })
    return summary


def print_chunk_summary(label: str, summary: List[dict]):
    print(f"\n[{label}] per-chunk motion energy (RAFT flow magnitude; mean +/- std across videos)")
    print(f"  {'chunk':>5} {'pixel_range':>12} {'interior_mean':>14} {'interior_std':>13} {'n':>4}   "
          f"{'boundary_mean':>14} {'boundary_std':>13} {'n':>4}")
    for row in summary:
        im = f"{row['interior_mean']:.4f}" if row['interior_mean'] is not None else "   n/a"
        istd = f"{row['interior_std']:.4f}" if row['interior_std'] is not None else "   n/a"
        bm = f"{row['boundary_mean']:.4f}" if row['boundary_mean'] is not None else "   n/a"
        bstd = f"{row['boundary_std']:.4f}" if row['boundary_std'] is not None else "   n/a"
        print(f"  {row['chunk']:>5} {str(row['pixel_range']):>12} {im:>14} {istd:>13} {row['interior_n_videos']:>4}   "
              f"{bm:>14} {bstd:>13} {row['boundary_n_videos']:>4}")

    interior_means = [row["interior_mean"] for row in summary if row["interior_mean"] is not None]
    if len(interior_means) >= 2:
        chunk0 = interior_means[0]
        rest = interior_means[1:]
        rest_mean = sum(rest) / len(rest)
        ratio = chunk0 / rest_mean if rest_mean > 0 else float("nan")
        print(f"  chunk0_interior_mean / mean(other_chunks_interior_mean) = {chunk0:.4f} / {rest_mean:.4f} = {ratio:.3f}")
        if ratio < 0.5:
            print("  -> chunk 0 moves noticeably less than later chunks (startup-stall pattern).")
        elif ratio > 1.5:
            print("  -> chunk 0 moves noticeably MORE than later chunks (decay-after-start pattern).")
        else:
            print("  -> chunk 0's motion level is roughly consistent with later chunks.")
