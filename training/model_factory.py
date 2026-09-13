from models.unet3d import UNet3D
from models.fno_wrapper import FNOWrapper
from models.ufno import UFNO3D
from models.loglo_fno import (
    ModulatedLOGLO_FNO, ModulatedLOGLO_FNO_Aux, VanillaLOGLO_FNO,
)


def create_model(model_cfg, model_type):
    if model_type == 'unet3d':
        return UNet3D(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_channels=model_cfg['hidden_channels'],
            depth=model_cfg.get('depth', 3),
            channel_multipliers=model_cfg.get('channel_multipliers', None),
        )
    if model_type == 'fno':
        return FNOWrapper(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
        )
    if model_type == 'ufno':
        return UFNO3D(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
            n_unet_layers=model_cfg['n_unet_layers'],
            lifting_channels=model_cfg.get('lifting_channels', 128),
            projection_channels=model_cfg.get('projection_channels', 128),
            unet_dropout=model_cfg.get('unet_dropout', 0.0),
        )
    if model_type == 'modulated_loglo':
        return ModulatedLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
            action_channels=model_cfg['action_channels'],
        )
    if model_type == 'modulated_loglo_aux':
        return ModulatedLOGLO_FNO_Aux(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
            action_channels=model_cfg['action_channels'],
            aux_hidden=model_cfg.get('aux_hidden', [128, 64]),
        )
    if model_type == 'vanilla_loglo':
        return VanillaLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
        )
    raise ValueError(f"Unknown model type: {model_type}")
