"""
EEG Peak Frequency Ratio Analysis

State-dependent frequency switching between φ and 2:1 dynamics.
"""

from .preprocessing import load_and_preprocess, get_valid_subjects
from .spectral import compute_psd, extract_peak_frequencies
from .analysis import compute_frequency_ratios, run_statistical_tests
from .visualization import plot_ratio_distribution, plot_state_comparison

__version__ = '1.0.0'
__author__ = 'Andrei Ursachi'
