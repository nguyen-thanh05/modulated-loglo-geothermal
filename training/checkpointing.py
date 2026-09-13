import os
from dataclasses import dataclass
from typing import Optional

import torch

from training.utils import capture_rng_states, restore_rng_states


@dataclass
class ResumeState:
    checkpoint: Optional[dict]
    optimizer_checkpoint: Optional[dict]
    start_epoch: int
    global_step: int
    wandb_run_id: Optional[str]


def _clean_state_dict(module):
    sd = module.state_dict()
    sd.pop('_metadata', None)
    return sd


class CheckpointManager:
    def __init__(self, checkpoint_config, *, resume_path, device):
        self.cfg = checkpoint_config
        self.resume_path = resume_path
        self.device = device

    @property
    def final_path(self):
        return self.cfg.final_path

    def final_exists(self):
        return os.path.isfile(self.cfg.final_path)

    def detect_resume(self):
        resume_ckpt = None
        optim_ckpt = None
        start_epoch = 0
        global_step = 0
        wandb_run_id = None

        if self.resume_path is not None and os.path.isfile(self.resume_path):
            print(f"[RESUME] Loading checkpoint: {self.resume_path}")
            resume_ckpt = torch.load(
                self.resume_path, map_location=self.device, weights_only=False)
            start_epoch = resume_ckpt['epoch'] + 1
            global_step = resume_ckpt['global_step']
            wandb_run_id = resume_ckpt.get('wandb_run_id')
            print(f"[RESUME] Will resume from epoch {start_epoch}, "
                  f"global_step {global_step}")

            if 'optimizer' in resume_ckpt:
                optim_ckpt = resume_ckpt
                print("[RESUME] Legacy checkpoint with embedded optimizer state.")
            elif os.path.isfile(self.cfg.optim_ckpt_path):
                optim_ckpt = torch.load(
                    self.cfg.optim_ckpt_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if optim_ckpt['epoch'] != resume_ckpt['epoch']:
                    print(f"[RESUME] Optimizer stale (epoch {optim_ckpt['epoch']+1} vs "
                          f"weights epoch {start_epoch}), restarting optimizer.")
                    optim_ckpt = None
                else:
                    print("[RESUME] Loaded matching optimizer state.")
            else:
                print("[RESUME] No optimizer checkpoint, restarting optimizer.")
        elif self.resume_path is not None:
            print(f"[RESUME] No checkpoint at {self.resume_path}, starting fresh.")

        return ResumeState(
            checkpoint=resume_ckpt,
            optimizer_checkpoint=optim_ckpt,
            start_epoch=start_epoch,
            global_step=global_step,
            wandb_run_id=wandb_run_id,
        )

    def apply_resume_state(
        self, resume_state, *, model, ema_model,
        optimizer, scheduler, loader_gen
    ):
        resume_ckpt = resume_state.checkpoint
        if resume_ckpt is None:
            return

        resume_ckpt['model'].pop('_metadata', None)
        resume_ckpt['ema_model'].pop('_metadata', None)
        model.load_state_dict(resume_ckpt['model'])
        ema_model.load_state_dict(resume_ckpt['ema_model'])
        restore_rng_states(resume_ckpt['rng_states'], loader_gen)

        optimizer_restored = False
        optim_ckpt = resume_state.optimizer_checkpoint
        if optim_ckpt is not None:
            optimizer.load_state_dict(optim_ckpt['optimizer'])
            scheduler.load_state_dict(optim_ckpt['scheduler'])
            optimizer_restored = True

        if not optimizer_restored and resume_state.global_step > 0:
            for _ in range(resume_state.global_step):
                scheduler.step()
            print(f"[RESUME] Scheduler advanced to step {resume_state.global_step}, "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")
        resume_state.checkpoint = None
        resume_state.optimizer_checkpoint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_resume(self, *, model, ema_model,
                    optimizer, scheduler, epoch, global_step, loader_gen,
                    wandb_run_id):
        self._save_weights_checkpoint(
            self.cfg.resume_ckpt_path,
            model=model,
            ema_model=ema_model,
            epoch=epoch,
            global_step=global_step,
            rng_states=capture_rng_states(loader_gen),
            wandb_run_id=wandb_run_id,
        )
        self._save_optimizer_state(
            self.cfg.optim_ckpt_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

    def save_final(self, *, model, ema_model):
        os.makedirs(os.path.dirname(self.cfg.final_path) or '.', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
        }, self.cfg.final_path)

    def remove_resume_files(self):
        for path in [self.cfg.resume_ckpt_path, self.cfg.optim_ckpt_path]:
            if os.path.isfile(path):
                os.remove(path)
                print(f"[CHECKPOINT] Training complete. Removed {os.path.basename(path)}")

    def _save_weights_checkpoint(self, path, *, model, ema_model,
                                 epoch, global_step, rng_states, wandb_run_id):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp_path = path + '.tmp'
        torch.save({
            'model': _clean_state_dict(model),
            'ema_model': _clean_state_dict(ema_model),
            'epoch': epoch,
            'global_step': global_step,
            'rng_states': rng_states,
            'wandb_run_id': wandb_run_id,
        }, tmp_path)
        os.replace(tmp_path, path)
        print(f"[CHECKPOINT] Weights saved: {path}  "
              f"(epoch={epoch+1}, step={global_step})")

    def _save_optimizer_state(self, path, *, optimizer, scheduler, epoch):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp_path = path + '.tmp'
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }, tmp_path)
        os.replace(tmp_path, path)
        print(f"[CHECKPOINT] Optimizer saved: {path}  (epoch={epoch+1})")
