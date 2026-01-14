"""
Configuration parameters for EEG frequency ratio analysis.
"""

import os
from pathlib import Path

# Golden ratio constant
PHI = (1 + 5**0.5) / 2  # ≈ 1.618033988749895

# Frequency band definitions (Hz)
BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

# Target ratios for hypothesis testing
TARGET_RATIOS = {
    'phi': PHI,           # 1.618 - anti-resonance
    'harmonic': 2.0,      # 2:1 - phase coupling
}

# Preprocessing parameters
PREPROC = {
    'bandpass': (1.0, 45.0),      # Hz
    'artifact_threshold': 100e-6,  # 100 μV in volts
    'reject_ratio': 0.5,           # Max fraction of rejected epochs
    'channels': ['Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Fz', 'Oz'],
}

# Spectral analysis parameters
SPECTRAL = {
    'window_sec': 2.0,        # Window length for Welch
    'overlap': 0.5,           # 50% overlap
    'freq_resolution': 0.5,   # Hz
}

# FOOOF parameters
FOOOF_SETTINGS = {
    'peak_width_limits': (1.0, 8.0),
    'max_n_peaks': 6,
    'min_peak_height': 0.1,
    'peak_threshold': 2.0,
    'aperiodic_mode': 'fixed',
}

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Create directories if needed
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# PhysioNet EEGMMIDB configuration
PHYSIONET = {
    'rest_runs': [1, 2],           # R01, R02 (eyes open/closed baseline)
    'task_runs': [4, 8, 12],       # R04, R08, R12 (motor imagery)
    'n_subjects': 109,
    'sfreq': 160,                  # Sampling frequency
}

# Statistical test parameters
STATS = {
    'alpha': 0.05,                 # Significance level
    'equivalence_margin': 0.15,    # TOST margin for φ equivalence
}
