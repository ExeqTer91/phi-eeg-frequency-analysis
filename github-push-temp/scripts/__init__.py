"""
EEG Phi Coupling Analysis Package

Golden Ratio Organization in Human EEG
"""

from .compute_pci import (
    compute_pci,
    compute_spectral_centroid,
    compute_peak_frequency,
    compute_convergence,
    compute_psd_welch,
    analyze_subject,
    PHI,
    EPSILON,
    THETA_BAND,
    ALPHA_BAND
)

__version__ = "1.0.0"
__author__ = "Andrei Condrea"
