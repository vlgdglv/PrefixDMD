import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle, ODERegressionPTDataset
from model import TeacherForcingDMD
import random
from collections import defaultdict
from utils.misc import (
    set_seed
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from collections import deque
from datetime import timedelta


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        # [FIX-1] Each rank must draw its OWN noise / tau. With set_seed(config.seed) on every rank,
        # all 8 GPUs sample identical tau, generation noise, critic noise and future padding each step,
        # so the DMD / critic estimators only see `batch_size` distinct noise draws instead of
        # batch_size * world_size. Self Forcing uses seed + rank. Everything that must be rank-consistent
        # (chunk_idx, on-policy plan, DFD switch) already uses its own step-seeded torch.Generator.
        if getattr(config, "per_rank_seed", True):
            set_seed(config.seed + global_rank)
        else:
            set_seed(config.seed)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.wandb_run_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        # assert config.trainer == "teacher_forcing_dmd"
        # self.model = ODERegression(config, device=self.device)
        model_cls = TeacherForcingDMD
            
        if self.is_main_process:
            print("[TFDMD Trainer] cls: ", model_cls)
        self.model = model_cls(config, device=self.device)
        
        
        # 3.  Load the model if provided
        if getattr(config, "generator_ckpt", False):
            if getattr(config, "starting_step", 0) != 0:
                self.step = config.starting_step
                print(f"Resuming from step: {self.step}!!")
            if self.is_main_process:
                print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        if getattr(config, "critic_ckpt", False):
            if self.is_main_process:
                print(f"Loading fake critic from {config.critic_ckpt}")
            state_dict = torch.load(config.critic_ckpt, map_location="cpu")
            if "critic" in state_dict:
                state_dict = state_dict["critic"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            else:
                raise NotImplementedError
            self.model.fake_score.load_state_dict(
                state_dict, strict=True
            )
            
        if getattr(config, "real_critic_ckpt", False):
            if self.is_main_process:
                print(f"Loading real critic from {config.real_critic_ckpt}")
            state_dict = torch.load(config.real_critic_ckpt, map_location="cpu")
            if "real_critic" in state_dict:
                state_dict = state_dict["real_critic"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            else:
                raise NotImplementedError
            self.model.real_score.load_state_dict(
                state_dict, strict=True
            )
        
        
        # [FIX-3] When a critic ckpt is loaded the fake critic used to be frozen during warmup.
        # If the critic input changes (e.g. window_chunks 1 -> 2) it must re-adapt: set fake_critic_warmup: true.
        self.warmup_fake_score = bool(getattr(config, "fake_critic_warmup",
                                              not bool(getattr(config, "critic_ckpt", None))))

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
    
        self.trainable_realscore = getattr(config, "trainable_realscore", True)
        if not self.trainable_realscore:
            print("NO update on real score.")

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.fake_critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_fake_critic if hasattr(config, "lr_fake_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )
        if self.trainable_realscore:
            self.real_critic_optimizer = torch.optim.AdamW(
                [param for param in self.model.real_score.parameters()
                if param.requires_grad],
                lr=config.lr_real_critic if hasattr(config, "lr_real_critic") else config.lr,
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=config.weight_decay
            )

        # Step 3: Initialize the dataloader
        data_store_ext = getattr(config, "data_store_ext", "lmdb")
        if data_store_ext == "lmdb":
            data_cls = ODERegressionLMDBDataset
        elif data_store_ext == "pt":
            data_cls = ODERegressionPTDataset
        else:
            data_cls = ODERegressionLMDBDataset
        dataset = data_cls(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"[Trainer] Dataset loaded. {len(dataset)} entries in total.")
            print(f"[Trainer] full config: ")
            print(config)
        # [FIX-2] do NOT reset self.step here: it silently discarded `starting_step`.
        # self.step is 0 unless starting_step was given together with generator_ckpt.
        self.critic_warmup_step = getattr(config, "critic_warmup_step", 100)
        self.save_critic = True

        # [GRAD-ACCUM] micro-batches per optimizer step (1 = original behaviour)
        
        self.grad_accum_steps = max(1, int(getattr(config, "grad_accum_steps", 1)))
        if self.is_main_process:
            print(f"[Trainer] grad_accum_steps={self.grad_accum_steps}, effective batch = "
                  f"{config.batch_size} x {self.world_size} x {self.grad_accum_steps} = "
                  f"{config.batch_size * self.world_size * self.grad_accum_steps}; "
                  f"per_rank_seed={getattr(config, 'per_rank_seed', True)}; "
                  f"fake_critic_warmup={self.warmup_fake_score}; start step={self.step}")
            if getattr(config, "use_tf_loss", False):
                engage_tf_step = getattr(config, "engage_tf_step", 0)
                print(f"TF Loss will be engaged at: {engage_tf_step}")
        ##############################################################################################################
        
        ##############################################################################################################
        # 6. Set up EMA parameter containers
        # No ema for now
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        # later
        # if (ema_weight is not None) and (ema_weight > 0.0):
        #     print(f"Setting up EMA with weight {ema_weight}")
        #     self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm = 10.0
        self.previous_time = None

        if getattr(config, "onpolicy_prob", 0.0) > 0 and getattr(config, "onpolicy_max_chunks", 1):
            print(f"Using on policy with probablity {config.onpolicy_prob} and max chunk {config.onpolicy_max_chunks}")
        if getattr(config, "dfd_prob", 0.0) > 0:
            print(f"Using on Data-Forcing with probablity {config.dfd_prob}")

    def _debug(self, msg):
        print(
            f"[rank {dist.get_rank()}] "
            f"[step {self.step}] {msg}",
            flush=True,
        )
        
    def save(self, save_real_score=True):
        print("Start gathering distributed model states...")
        # [FIX-5] guard: EMA may not exist (ema_weight=0, or no generator update since ema_start_step);
        # the old code crashed with generator_ema=None. model.pt keeps the old semantics
        # (EMA if it exists, raw generator otherwise) so eval scripts behave as before.
        use_ema = (self.generator_ema is not None) and (self.config.ema_start_step < self.step)
        raw_generator_state_dict = None
        if use_ema:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
            if getattr(self.config, "save_raw_generator", False):
                raw_generator_state_dict = {"generator": fsdp_state_dict(self.model.generator)}
        else:
            state_dict = {
                "generator": fsdp_state_dict(self.model.generator),
            }

        if self.save_critic:
            critic_state_dict = {"critic": fsdp_state_dict(self.model.fake_score)}
        if save_real_score:
            real_critic_state_dict = {"real_critic": fsdp_state_dict(self.model.real_score)}

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}",  "model.pt"))
            if raw_generator_state_dict is not None:                           # [FIX-5] for resuming training
                torch.save(raw_generator_state_dict, os.path.join(self.output_path,
                           f"checkpoint_model_{self.step:06d}", "generator_raw.pt"))

            if self.save_critic:
                torch.save(critic_state_dict, os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}", "critic_model.pt"))
            if save_real_score:
                torch.save(real_critic_state_dict, os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}", "real_critic_model.pt"))
                
    
    def _chunk_plan(self, ode_latent, weight_key):
        total = ode_latent.shape[2] // self.model.num_frame_per_block   # ode_latent: [B, T_traj, F, C, H, W]
        if not getattr(self.config, "anchor_first_chunk", False):
            return [(None, 1.0)]
        idx = 1 + self.model.sample_chunk_idx(self.step, total - 1)
        return [(0, float(getattr(self.config, weight_key, 1.0))), (idx, 1.0)]

    # [GRAD-ACCUM] fetch K micro-batches once per step; shared by real critic / fake critic / generator
    def _get_micro_batches(self):
        micro_batches = []
        for _ in range(self.grad_accum_steps):
            batch = next(self.dataloader)
            ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(text_prompts=batch["prompts"])
            micro_batches.append((ode_latent, conditional_dict))
        return micro_batches

    def train_one_step_generator(self, micro_batches, loss_scale=1.0):
        self.model.generator.train()
        self.model.real_score.eval()
        self.model.fake_score.eval()

        accum = len(micro_batches)
        self.generator_optimizer.zero_grad(set_to_none=True)
        generator_loss = 0.0
        for ode_latent, conditional_dict in micro_batches:                  # [GRAD-ACCUM]
                
            for idx, w in self._chunk_plan(ode_latent, "anchor_weight_gen"):
                loss, log_dict = self.model.generator_loss(
                    ode_latent=ode_latent, conditional_dict=conditional_dict,
                    unconditional_dict=self.unconditional_dict, global_step=self.step, chunk_idx=idx)
                
                # Normal backward
                (loss * w / accum).backward()
                generator_loss = generator_loss + loss.detach() / accum
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)   # clip once, after accumulation
        self.generator_optimizer.step()

        # Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb_loss_dict = {
                    "generator_loss": generator_loss.item(),
                    "generator_grad_norm": generator_grad_norm.item(),
                    "chunk_idx": log_dict["chunk_idx"],
                }
                wandb.log(wandb_loss_dict, step=self.step)
                if getattr(self.config, "use_tf_loss", False) and self.step > getattr(self.config, "engage_tf_step", 0):
                    tf_loss = log_dict["tf_loss"]
                    dmd_loss = log_dict["dmd_loss"]
                    print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item(),
                     "dmd_loss", dmd_loss.item(), "tf_loss: ", tf_loss.item(), "chunk_idx: ", log_dict["chunk_idx"])
                else:
                    print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item(), "chunk_idx: ", log_dict["chunk_idx"])
            elif self.step % self.config.console_print_interval == 0:
                if getattr(self.config, "use_tf_loss", False) and self.step > getattr(self.config, "engage_tf_step", 0):
                    tf_loss = log_dict["tf_loss"]
                    dmd_loss = log_dict["dmd_loss"]
                    print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item(),
                     "dmd_loss", dmd_loss.item(), "tf_loss: ", tf_loss.item(), "chunk_idx: ", log_dict["chunk_idx"])
                else:
                    print("Step: ", self.step, ", generator_loss: ", generator_loss.item(), "generator_grad_norm: ", generator_grad_norm.item(), "chunk_idx: ", log_dict["chunk_idx"])
            else:
                pass

    def train_one_step_real_critic(self, micro_batches, loss_scale=1.0):
        self.model.real_score.train()
        self.model.generator.eval()
        self.model.fake_score.eval()
        self.real_critic_optimizer.zero_grad(set_to_none=True)

        p = getattr(self.config, "real_cond_dropout", 0.0)
        accum = len(micro_batches)
        critic_loss = 0.0
        for ode_latent, conditional_dict in micro_batches:                     # [GRAD-ACCUM]
            # [FIX-6] the dropped-condition embedding is configurable (see self.dropout_dict in train())
            if p > 0 and random.random() < p:
                conditional_dict = self.dropout_dict
            for idx, w in self._chunk_plan(ode_latent, "anchor_weight_critic"):   # [FIX-4] critic weight key
                loss, critic_log_dict = self.model.real_critic_loss(
                    ode_latent=ode_latent, conditional_dict=conditional_dict,
                    global_step=self.step, chunk_idx=idx)
                (loss * w / accum).backward()
                critic_loss = critic_loss + loss.detach() / accum
        critic_grad_norm = self.model.real_score.clip_grad_norm_(self.max_grad_norm)
        self.real_critic_optimizer.step()

        critic_log_dict.update({
            "real_critic_loss": critic_loss,
            "real_critic_grad_norm": critic_grad_norm
        })

        # Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb_loss_dict = {
                    "real_critic_loss": critic_loss.item(),
                    "real_critic_grad_norm": critic_grad_norm.item(),
                    "chunk_idx": critic_log_dict["chunk_idx"],
                }
                wandb.log(wandb_loss_dict, step=self.step)
                if self.step % 25 == 0:
                    print("Step: ", self.step, ", real_critic_loss: ", critic_loss.item(), "real_critic_grad_norm: ", critic_grad_norm.item(),  "chunk_idx: ", critic_log_dict["chunk_idx"])
            elif self.step % self.config.console_print_interval == 0:
                print("Step: ", self.step, ", real_critic_loss: ", critic_loss.item(), "real_critic_grad_norm: ", critic_grad_norm.item(), "chunk_idx: ", critic_log_dict["chunk_idx"])
            else:
                pass

    def train_one_step_fake_critic(self, micro_batches, loss_scale=1.0):
        self.model.fake_score.train()
        self.model.generator.eval()
        self.model.real_score.eval()
        self.fake_critic_optimizer.zero_grad(set_to_none=True)

        accum = len(micro_batches)
        critic_loss = 0.0
        for ode_latent, conditional_dict in micro_batches:                     # [GRAD-ACCUM]
            for idx, w in self._chunk_plan(ode_latent, "anchor_weight_critic"):   # [FIX-4] critic weight key
                loss, critic_log_dict = self.model.fake_critic_loss(
                    ode_latent=ode_latent, conditional_dict=conditional_dict,
                    global_step=self.step, chunk_idx=idx)
                (loss * w / accum).backward()
                critic_loss = critic_loss + loss.detach() / accum
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm)
        self.fake_critic_optimizer.step()

        critic_log_dict.update({
            "fake_critic_loss": critic_loss,
            "fake_critic_grad_norm": critic_grad_norm
        })

        # Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb_loss_dict = {
                    "fake_critic_loss": critic_loss.item(),
                    "fake_critic_grad_norm": critic_grad_norm.item(),
                    "chunk_idx": critic_log_dict["chunk_idx"],
                }
                wandb.log(wandb_loss_dict, step=self.step)
                if self.step % 25 == 0:
                    print("Step: ", self.step, ", fake_critic_loss: ", critic_loss.item(), "fake_critic_grad_norm: ", critic_grad_norm.item(), "chunk_idx: ", critic_log_dict["chunk_idx"])
            elif self.step % self.config.console_print_interval == 0:
                print("Step: ", self.step, ", fake_critic_loss: ", critic_loss.item(), "fake_critic_grad_norm: ", critic_grad_norm.item(), "chunk_idx: ", critic_log_dict["chunk_idx"])
            else:
                pass

    def train(self):
        max_steps = int(getattr(self.config, "max_steps", -1))
        batch_size = int(getattr(self.config, "batch_size", 1))
        if self.is_main_process:
            print("Training begin.")

        with torch.no_grad():
            self.unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size
            )
            self.model.unconditional_dict = self.unconditional_dict
            # [FIX-6] which embedding real_cond_dropout trains as "unconditional":
            #   "negative" (default, = old behaviour): trains exactly the uncond branch used by CFG in generator_loss
            #   "empty": trains the empty-prompt branch (keeps the negative prompt's anti-static push untouched)
            if getattr(self.config, "real_cond_dropout_prompt", "negative") == "empty":
                self.dropout_dict = self.model.text_encoder(text_prompts=[""] * batch_size)
            else:
                self.dropout_dict = self.unconditional_dict
        
        
        if self.is_main_process:
            eta_times = deque(maxlen=50)
            eta_last_time = time.perf_counter()
            
        while max_steps < 0 or self.step < max_steps:
            critic_warmup = self.step < self.critic_warmup_step
            do_train_generator = not critic_warmup and self.step > 0 and self.step % self.config.dfake_gen_update_ratio == 0
            do_train_fake_critic = (critic_warmup and self.warmup_fake_score) or (not critic_warmup)
            do_train_real_critic = self.trainable_realscore and self.step > 0 and self.step % self.config.dreal_critic_update_ratio == 0
            
            # [GRAD-ACCUM] K micro-batches per step (K=1 reproduces the original single-batch behaviour)
            micro_batches = self._get_micro_batches()

            if do_train_real_critic:
                self.train_one_step_real_critic(micro_batches)

            if do_train_fake_critic:
                self.train_one_step_fake_critic(micro_batches)

            # Train the generator
            if do_train_generator:
                self.train_one_step_generator(micro_batches)
                if self.config.ema_weight > 0 and self.step >= self.config.ema_start_step:
                    if self.generator_ema is None:
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                    else:
                        self.generator_ema.update(self.model.generator)

            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step > 0:
                self.save()
                torch.cuda.empty_cache()
            
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                # barrier()
                
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
            if self.is_main_process:
                now = time.perf_counter()
                eta_times.append(now - eta_last_time)
                eta_last_time = now

                if max_steps > 0 and (self.step % 10 == 0 or self.step == max_steps):
                    avg_step_time = sum(eta_times) / len(eta_times)
                    remaining_steps = max(0, max_steps - self.step)
                    eta_seconds = remaining_steps * avg_step_time
                    print(
                        f"[ETA] Step {self.step}/{max_steps} "
                        f"({100 * self.step / max_steps:.1f}%) | "
                        f"{avg_step_time:.2f}s/step | "
                        f"Remaining: {timedelta(seconds=int(eta_seconds))}",
                        flush=True,
                    )
        self.save(save_real_score=True)