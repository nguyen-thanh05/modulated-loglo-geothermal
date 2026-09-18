import matplotlib.pyplot as plt
import torch
import wandb

from training.constants import CHANNEL_NAMES, WELL_COORDS
from training.model_adapters import split_model_output
from training.physics import compute_mean_pressure_loss


class TrainingLogger:
    def __init__(self, run_config):
        self.cfg = run_config

    @property
    def writer(self):
        return self.cfg.logging.writer

    def setup(self, *, model, wandb_run_id):
        if not self.writer:
            return wandb_run_id

        model_type = self.cfg.model.type
        run_tag = self.cfg.logging.run_tag
        run_name = f"{model_type}{run_tag}-hetero-seed{self.cfg.seed}"
        init_kwargs = dict(
            project=self.cfg.logging.wandb_project or 'heterogeneous',
            entity=self.cfg.logging.wandb_entity,
            name=run_name,
            config=self.cfg.raw,
            tags=[model_type, 'hetero'],
            group=model_type,
        )
        if wandb_run_id is not None:
            init_kwargs['id'] = wandb_run_id
            init_kwargs['resume'] = 'must'
        wandb.init(**init_kwargs)

        if wandb_run_id is None:
            wandb.config.update({"seed": self.cfg.seed})
            if self.cfg.logging.watch_model:
                wandb.watch(model, log="all", log_freq=self.cfg.logging.watch_freq)
        return wandb.run.id

    def log_training_scalars(self, *, one_step_loss, loss_l2_rel,
                             total_loss, loss_pf, k, lr, global_step):
        wandb.log({
            'Loss/MSE': one_step_loss.loss_mse.item(),
            'Loss/H1': one_step_loss.loss_h1.item(),
            'Loss/MeanPressure': one_step_loss.loss_mean_pressure.item(),
            'Loss/L2_Rel': loss_l2_rel.item(),
            'Loss/MBE': one_step_loss.loss_mbe.item(),
            'Loss/Spectral_Low': one_step_loss.spectral_bands[0].item(),
            'Loss/Spectral_Mid': one_step_loss.spectral_bands[1].item(),
            'Loss/Spectral_High': one_step_loss.spectral_bands[2].item(),
            'Loss/Aux': one_step_loss.loss_aux.item(),
            'Loss/Aux_BHP': one_step_loss.loss_aux_bhp.item(),
            'Loss/Aux_Energy': one_step_loss.loss_aux_energy.item(),
            'Loss/Total': total_loss.item(),
            'Loss/Pushforward': loss_pf.item(),
            'Training/k': k,
            'Training/lr': lr,
        }, step=global_step)

    def run_validation(self, *, model, test_loader, adapter,
                       loss_computer, device, global_step):
        model.eval()
        with torch.no_grad():
            _, _, _, val_y_t, val_y_tp1, val_act_t, val_static, _, val_aux_tp1 = next(
                iter(test_loader))
            val_static = val_static.to(device)
            val_y_t = val_y_t.to(device)
            val_y_tp1 = val_y_tp1.to(device)
            val_act_t = val_act_t.to(device)
            val_aux_tp1 = val_aux_tp1.to(device)

            for well in WELL_COORDS:
                val_act_t[:, 0:2, well[0], well[1]] = 0.

            val_input = adapter.build_model_input(val_y_t, val_act_t, val_static)
            val_pred, val_pred_aux = split_model_output(
                adapter.forward(model, val_input))

            loss_mse_val = loss_computer.mse_fn(val_pred, val_y_tp1)
            loss_h1_val = loss_computer.calculate_weighted_h1_loss(val_pred, val_y_tp1)
            loss_mean_pressure_val = compute_mean_pressure_loss(val_pred, val_y_tp1)
            loss_l2_rel_val = loss_computer.l2_relative(val_pred, val_y_tp1)
            _, loss_aux_bhp_val, loss_aux_energy_val = (
                loss_computer.compute_aux_components(val_pred_aux, val_aux_tp1)
            )
            loss_aux_val = (
                loss_computer.mse_fn(val_pred_aux, val_aux_tp1)
                if val_pred_aux is not None
                else torch.tensor(0.0, device=device)
            )

            val_log = {
                'Val_Loss/MSE': loss_mse_val.item(),
                'Val_Loss/H1': loss_h1_val.item(),
                'Val_Loss/MeanPressure': loss_mean_pressure_val.item(),
                'Val_Loss/L2_Rel': loss_l2_rel_val.item(),
                'Val_Loss/Aux': loss_aux_val.item(),
                'Val_Loss/Aux_BHP': loss_aux_bhp_val.item(),
                'Val_Loss/Aux_Energy': loss_aux_energy_val.item(),
            }

            for i, ch_name in enumerate(CHANNEL_NAMES):
                truth = val_y_tp1[0, i, 10, :, :].cpu().numpy()
                pred_np = val_pred[0, i, 10, :, :].cpu().numpy()
                error = pred_np - truth
                vmax = max(abs(error.min()), abs(error.max())) or 1.0

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].imshow(truth)
                axes[0].set_title(f'True {ch_name}')
                axes[0].axis('off')
                axes[1].imshow(pred_np)
                axes[1].set_title(f'Pred {ch_name}')
                axes[1].axis('off')
                im = axes[2].imshow(error, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[2].set_title(f'Error {ch_name}')
                axes[2].axis('off')
                fig.colorbar(im, ax=axes[2], fraction=0.046)
                plt.tight_layout()
                val_log[f'Val_Image/{ch_name}'] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(val_log, step=global_step)

    def log_final_artifact(self, final_path):
        if not self.writer:
            return
        try:
            art = wandb.Artifact(
                f"{self.cfg.model.type}{self.cfg.logging.run_tag}-hetero-final",
                type="model",
            )
            art.add_file(final_path)
            wandb.log_artifact(art)
        except Exception as e:
            print(f"[WARNING] Final artifact upload failed: {e}")

    def finish(self, exit_code=None):
        if not self.writer:
            return
        if exit_code is None:
            wandb.finish()
        else:
            wandb.finish(exit_code=exit_code)
