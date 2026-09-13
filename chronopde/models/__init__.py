"""Trainable ChronoPDE models and controlled baselines."""

from chronopde.models.common import trainable_parameter_count
from chronopde.models.ct_dct import CosineSpectralConv2d, DCTContinuousVectorField
from chronopde.models.ct_fno import FFTContinuousVectorField, FiLMConditioner
from chronopde.models.fno import FNOAutoregressive, FourierSpectralConv2d
from chronopde.models.unet import UNetAutoregressive

__all__ = [
    "CosineSpectralConv2d",
    "DCTContinuousVectorField",
    "FFTContinuousVectorField",
    "FNOAutoregressive",
    "FiLMConditioner",
    "FourierSpectralConv2d",
    "UNetAutoregressive",
    "trainable_parameter_count",
]
