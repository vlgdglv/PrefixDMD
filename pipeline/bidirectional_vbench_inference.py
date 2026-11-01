"""
Raw (base) bidirectional Wan2.1 T2V inference pipeline, used to generate VBench
evaluation videos.

This mirrors the model configuration used by the ODE-trajectory generation
script (48-step flow-matching sampling, timestep_shift=5.0, batched CFG) --
i.e. the *teacher* distribution our causal/TF-DMD students are distilled
from -- rather than the UniPC/DPM++ solvers used by
`pipeline/bidirectional_diffusion_inference.py`. It intentionally does not
use any causal KV-cache machinery: every frame attends to every other frame
every step (standard bidirectional diffusion).
"""
from typing import List, Optional

import torch
from tqdm import tqdm

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalVBenchInferencePipeline(torch.nn.Module):
    def __init__(
        self,
        device,
        model_name: str = "Wan2.1-T2V-1.3B",
        timestep_shift: float = 5.0,
        num_inference_steps: int = 48,
        guidance_scale: float = 6.0,
        negative_prompt: str = (
            '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，'
            '整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，'
            '画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，'
            '静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
        ),
        latent_shape=(21, 16, 60, 104),  # (num_frames, num_channels, height, width)
    ):
        super().__init__()
        self.device = device
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.num_frames, self.num_channels, self.height, self.width = latent_shape

        self.generator = WanDiffusionWrapper(
            model_name=model_name, timestep_shift=timestep_shift, is_causal=False,
        )
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0)

        self._unconditional_dict = None
        # Dummy buffer so `.to(dtype=...)` casts are observable afterward (used
        # to sample noise in the same dtype as the rest of the pipeline,
        # without relying on any private torch API).
        self.register_buffer("_dtype_marker", torch.zeros(1), persistent=False)

    @property
    def dtype(self):
        return self._dtype_marker.dtype

    def to(self, *args, **kwargs):
        """
        NOTE: `device` is fixed at construction time (used for noise sampling)
        and is NOT re-inferred from `.to(...)` calls here -- construct with
        the correct device and only use `.to()` afterward for dtype casts.
        """
        super().to(*args, **kwargs)
        self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        return self

    def load_generator_checkpoint(self, checkpoint_path, use_ema=False, key=None):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = "generator_ema" if (use_ema and "generator_ema" in state_dict) else "generator" if key is None else key
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

    @property
    def unconditional_dict(self):
        if self._unconditional_dict is None:
            with torch.no_grad():
                self._unconditional_dict = self.text_encoder(text_prompts=[self.negative_prompt])
        return self._unconditional_dict

    @torch.no_grad()
    def inference(
        self,
        text_prompts: List[str],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False,
        show_progress: bool = True,
    ):
        """
        Inputs:
            text_prompts: list of strings, batch dimension.
            noise: optional pre-sampled noise [B, F, C, H, W]; sampled fresh
                (using `generator`, if given) when omitted.
        Outputs:
            video: [B, F, H, W, 3] tensor in [0, 1] (decoded, pixel-space).
        """
        batch_size = len(text_prompts)
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = {
            k: v.repeat(batch_size, *([1] * (v.ndim - 1))) for k, v in self.unconditional_dict.items()
        }

        if noise is None:
            noise = torch.randn(
                [batch_size, self.num_frames, self.num_channels, self.height, self.width],
                device=self.device, dtype=self.dtype, generator=generator,
            )
        latents = noise

        for t in tqdm(self.scheduler.timesteps, disable=not show_progress):
            timestep = t * torch.ones([batch_size, self.num_frames], device=self.device, dtype=torch.float32)

            # CFG batching: [uncond, cond] in one forward pass.
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
            # Unlike `scheduler.add_noise`, `.step()` has no internal dtype
            # cast; under mixed precision `sample + model_output * (...)`
            # silently upcasts to float32, which then breaks the next forward
            # pass against lower-precision weights. Force it back.
            latents = latents.to(self.dtype)

        video = self.vae.decode_to_pixel(latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        if return_latents:
            return video, latents
        return video

    @torch.no_grad()
    def inference_mixedts(
        self,
        text_prompts: List[str],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False,
        show_progress: bool = True,
        frame_end: int = 21,
    ):
        """
        Inputs:
            text_prompts: list of strings, batch dimension.
            noise: optional pre-sampled noise [B, F, C, H, W]; sampled fresh
                (using `generator`, if given) when omitted.
        Outputs:
            video: [B, F, H, W, 3] tensor in [0, 1] (decoded, pixel-space).
        """
        s
        batch_size = len(text_prompts)
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = {
            k: v.repeat(batch_size, *([1] * (v.ndim - 1))) for k, v in self.unconditional_dict.items()
        }

        if noise is None:
            noise = torch.randn(
                [batch_size, self.num_frames, self.num_channels, self.height, self.width],
                device=self.device, dtype=self.dtype, generator=generator,
            )
        latents = noise

        for t in tqdm(self.scheduler.timesteps, disable=not show_progress):
            latents[:, frame_end:] = torch.randn_like(latents[:, frame_end:],  device=self.device, dtype=self.dtype, generator=generator)

            timestep = torch.cat([
                torch.full((1, frame_end), t,  device=self.device, dtype=torch.float32), 
                torch.full((1, 21 - frame_end), 1000.,  device=self.device, dtype=torch.float32)
                ],dim=1)

            # CFG batching: [uncond, cond] in one forward pass.
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

            latents[:, :frame_end] = self.scheduler.step(
                flow_pred[:, :frame_end].flatten(0, 1),
                timestep[:, :frame_end].flatten(0, 1),
                latents[:, :frame_end].flatten(0, 1),
            ).unflatten(0, flow_pred[:, :frame_end].shape[:2])
            # Unlike `scheduler.add_noise`, `.step()` has no internal dtype
            # cast; under mixed precision `sample + model_output * (...)`
            # silently upcasts to float32, which then breaks the next forward
            # pass against lower-precision weights. Force it back.
            latents = latents.to(self.dtype)

        latents = latents[:, :frame_end]
        video = self.vae.decode_to_pixel(latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        if return_latents:
            return video, latents
        return video


    @torch.no_grad()
    def inference_contexted_chunk(
        self,
        text_prompts: List[str],
        context_latent: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False,
        show_progress: bool = True,
        frame_num: int = 21,
        frame_per_chunk: int = 3,
    ):
        """
        Inputs:
            text_prompts: list of strings, batch dimension.
            context_latent: clean latent from some video.
        Outputs:
            video: [B, F, H, W, 3] tensor in [0, 1] (decoded, pixel-space).
        """
        self.generator.uniform_timestep = False
        batch_size = len(text_prompts)
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = {
            k: v.repeat(batch_size, *([1] * (v.ndim - 1))) for k, v in self.unconditional_dict.items()
        }

        assert context_latent is not None

        output_latent = context_latent.clone()
        chunk_num = frame_num // frame_per_chunk
        for cid in range(chunk_num):
            latents = context_latent.clone()
            frame_start = cid * frame_per_chunk
            frame_end = (cid + 1) * frame_per_chunk

            latents[:, frame_start:] = torch.randn_like(latents[:, frame_start:],  device=self.device, dtype=self.dtype, generator=generator)
            for t in tqdm(self.scheduler.timesteps, disable=not show_progress):

                timestep = torch.cat([
                    torch.full((1, frame_start), 0,  device=self.device, dtype=torch.float32), 
                    torch.full((1, frame_per_chunk), t,  device=self.device, dtype=torch.float32), 
                    torch.full((1, 21 - frame_end), 1000.,  device=self.device, dtype=torch.float32)
                    ],dim=1)

                # CFG batching: [uncond, cond] in one forward pass.
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

                latents[:, frame_start:frame_end] = self.scheduler.step(
                    flow_pred[:, frame_start:frame_end].flatten(0, 1),
                    timestep[:, frame_start:frame_end].flatten(0, 1),
                    latents[:, frame_start:frame_end].flatten(0, 1),
                ).unflatten(0, flow_pred[:, frame_start:frame_end].shape[:2])
                # Unlike `scheduler.add_noise`, `.step()` has no internal dtype
                # cast; under mixed precision `sample + model_output * (...)`
                # silently upcasts to float32, which then breaks the next forward
                # pass against lower-precision weights. Force it back.
                latents = latents.to(self.dtype)

            output_latent[:, frame_start:frame_end] = latents[:, frame_start:frame_end]

        video = self.vae.decode_to_pixel(output_latent, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        if return_latents:
            return video, output_latent
        return video