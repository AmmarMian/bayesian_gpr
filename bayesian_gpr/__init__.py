"""bayesian_gpr — GPR physics tools for Bayesian MCMC inversion.

Provides forward simulator, scene builder, clutter generator, and Huber norm
function. The MCMC sampler, priors, and likelihood are the caller's responsibility.
"""
from .scene import Atom, Scene, SceneGrid, SoilParams, WaveletParams
from .forward import render_atoms, render_via_conv, ricker
from .huber import huber, huber_total
from . import presets

__all__ = [
    "Atom",
    "Scene",
    "SceneGrid",
    "SoilParams",
    "WaveletParams",
    "render_atoms",
    "render_via_conv",
    "ricker",
    "huber",
    "huber_total",
    "presets",
]
