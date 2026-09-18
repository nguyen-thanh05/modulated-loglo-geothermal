import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    path: str


@dataclass
class ModelConfig:
    raw: Dict[str, Any]

    @property
    def type(self):
        return self.raw['type']


@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    num_epochs: int
    test_batch_size: int
    weight_decay: float
    ema_decay: float
    log_every: int
    grad_clip_norm: float
    noise_alpha: Any
    noise_probability: float
    pushforward_k_max: int
    pushforward_k_step_interval: int
    warmup_steps: int
    cooldown_steps: int
    min_lr: float
    use_pushforward: bool
    epochs_per_run: int
    aux_start_step: int


@dataclass
class LossConfig:
    use_mse: bool
    mse_weight: float
    use_h1: bool
    h1_weight: float
    channel_weights: Any
    use_mbe: bool
    mbe_weight: float
    use_spectral: bool
    spectral_weight: float
    spectral_iLow: int
    spectral_iHigh: int
    aux_weight: float
    use_mean_pressure: bool
    mean_pressure_weight: float


@dataclass
class LoggingConfig:
    writer: bool
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    watch_model: bool
    watch_freq: int
    log_scalar_every: int
    run_tag: str


@dataclass
class CheckpointConfig:
    running_dir: str
    final_path: str
    resume_ckpt_path: str
    optim_ckpt_path: str
    save_every: int


@dataclass
class RunConfig:
    raw: Dict[str, Any]
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    logging: LoggingConfig
    checkpoints: CheckpointConfig


def build_run_config(cfg, args):
    training_root = cfg['training']
    log_every = training_root['log_every']
    num_epochs = training_root['num_epochs']

    running_dir = cfg['checkpoints']['running_dir']
    resume_ckpt_path = cfg['checkpoints'].get(
        'resume_path', os.path.join(running_dir, 'resume_checkpoint.pt'))

    return RunConfig(
        raw=cfg,
        seed=args.seed,
        data=DataConfig(path=cfg['data']['path']),
        model=ModelConfig(raw=cfg['model']),
        training=TrainingConfig(
            batch_size=training_root['batch_size'],
            lr=training_root['lr'],
            num_epochs=num_epochs,
            test_batch_size=training_root['test_batch_size'],
            weight_decay=training_root['weight_decay'],
            ema_decay=training_root['ema_decay'],
            log_every=log_every,
            grad_clip_norm=training_root['grad_clip_norm'],
            noise_alpha=training_root.get(
                'noise_alpha', [0.0025, 0.0025, 0.025, 0.025]),
            noise_probability=training_root.get('noise_probability', 0.8),
            pushforward_k_max=training_root.get('pushforward_k_max', 5),
            pushforward_k_step_interval=training_root.get(
                'pushforward_k_step_interval', 300),
            warmup_steps=training_root.get('warmup_steps', 0),
            cooldown_steps=training_root.get('cooldown_steps', 0),
            min_lr=training_root.get('min_lr', 1e-5),
            use_pushforward=training_root.get('use_pushforward', True),
            epochs_per_run=training_root.get('epochs_per_run', num_epochs),
            aux_start_step=training_root.get('aux_start_step', 0),
        ),
        loss=LossConfig(
            use_mse=cfg['loss'].get('use_mse', True),
            mse_weight=cfg['loss'].get('mse_weight', 1.0),
            use_h1=cfg['loss'].get('use_h1', True),
            h1_weight=cfg['loss'].get('h1_weight', 1.0),
            channel_weights=cfg['loss']['channel_weights'],
            use_mbe=cfg['loss'].get('use_mbe', True),
            mbe_weight=cfg['loss'].get('mbe_weight', 3.0),
            use_spectral=cfg['loss'].get('use_spectral', True),
            spectral_weight=cfg['loss'].get('spectral_weight', 0.0),
            spectral_iLow=cfg['loss'].get('spectral_iLow', 2),
            spectral_iHigh=cfg['loss'].get('spectral_iHigh', 10),
            aux_weight=cfg['loss'].get('aux_weight', 0.0),
            use_mean_pressure=cfg['loss'].get('use_mean_pressure', False),
            mean_pressure_weight=cfg['loss'].get('mean_pressure_weight', 0.0),
        ),
        logging=LoggingConfig(
            writer=cfg['logging']['writer'],
            wandb_project=cfg['logging'].get('wandb_project', None),
            wandb_entity=cfg['logging'].get('wandb_entity', None),
            watch_model=cfg['logging'].get('watch_model', False),
            watch_freq=cfg['logging'].get('watch_freq', 1000),
            log_scalar_every=cfg['logging'].get('log_scalar_every', 10),
            run_tag=cfg['logging'].get('run_tag', ''),
        ),
        checkpoints=CheckpointConfig(
            running_dir=running_dir,
            final_path=cfg['checkpoints']['final_path'],
            resume_ckpt_path=resume_ckpt_path,
            optim_ckpt_path=resume_ckpt_path.replace('.pt', '_optim.pt'),
            save_every=cfg['checkpoints'].get('save_every', log_every * 5),
        ),
    )
