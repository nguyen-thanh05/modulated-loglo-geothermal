"""Model package exports.

Heavy modules are imported lazily so optional dependencies stay unloaded
until a specific architecture is requested.
"""

__all__ = [
    "ModulatedLOGLO_FNO",
    "ModulatedLOGLO_FNO_Aux",
    "VanillaLOGLO_FNO",
    "FNOWrapper",
    "UFNO3D",
    "UNet3D",
]


def __getattr__(name):
    if name == "ModulatedLOGLO_FNO":
        from .loglo_fno import ModulatedLOGLO_FNO
        return ModulatedLOGLO_FNO
    if name == "ModulatedLOGLO_FNO_Aux":
        from .loglo_fno import ModulatedLOGLO_FNO_Aux
        return ModulatedLOGLO_FNO_Aux
    if name == "VanillaLOGLO_FNO":
        from .loglo_fno import VanillaLOGLO_FNO
        return VanillaLOGLO_FNO
    if name == "FNOWrapper":
        from .fno_wrapper import FNOWrapper
        return FNOWrapper
    if name == "UFNO3D":
        from .ufno import UFNO3D
        return UFNO3D
    if name == "UNet3D":
        from .unet3d import UNet3D
        return UNet3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
