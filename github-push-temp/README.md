# Golden Ratio Organization in Human EEG

[![DOI](https://img.shields.io/badge/Manuscript-Frontiers%201781338-blue)](https://doi.org/PENDING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Validation code and data for:** "Golden Ratio Organization in Human EEG is Associated with Theta-Alpha Frequency Convergence: A Multi-Dataset Validation Study"

Manuscript ID: 1781338 | Frontiers in Human Neuroscience

## Summary

This repository contains analysis code demonstrating that human EEG theta-alpha frequency organization clusters near the golden ratio (φ ≈ 1.618). Using the Phi Coupling Index (PCI), we analyzed resting-state EEG from **N=394 subjects** across three independent datasets.

## Key Results

| Dataset | N | Fs (Hz) | r (PCI-Conv) | % φ-organized | Mean α/θ ratio |
|---------|---|---------|--------------|---------------|----------------|
| PhysioNet EEGBCI | 109 | 160 | 0.674*** | 87.2% | 1.678 |
| OpenNeuro ds003969 | 78 | 500 | 0.622*** | 84.6% | 1.606 |
| LEMON Mind-Brain-Body | 207 | 2500 | 0.549*** | 84.5% | 1.669 |
| **Combined** | **394** | - | **0.576*** | **85.3%** | **1.659** |

***p < 10⁻⁹

### Critical Findings

**Frontal Theta Validation:**
- Frontal theta PCI-Convergence: **r = 0.874** (p < 10⁻⁶⁵)
- Posterior theta PCI-Convergence: r = 0.549
- Effect is STRONGER with frontal theta, ruling out volume conduction artifacts

**Null Model Validation:**
- Null model (destroying structure): r = 0.35 (SD = 0.05)
- Observed correlation: r = 0.576
- Exceeds null by **>4 SD** (z = 4.52, p < 0.0001)

## Metrics

### Phi Coupling Index (PCI)
```
PCI = log((|R - 2.0| + ε) / (|R - φ| + ε))
```
Where R = f_alpha / f_theta, φ = 1.618034, ε = 0.1

- PCI > 0: Closer to φ than to 2:1 harmonic (φ-organized)
- PCI < 0: Closer to 2:1 harmonic than to φ

### Spectral Centroid
```
f_centroid = Σ(f × PSD(f)) / Σ(PSD(f))
```
Computed within theta (4-8 Hz) and alpha (8-13 Hz) bands using Welch's method (4-second Hann windows, 50% overlap).

### Theta-Alpha Convergence
```
Convergence = 1 / |f_alpha - f_theta|
```
Higher values indicate frequencies approaching the ~8 Hz boundary.

## Repository Structure

```
eeg-phi-coupling/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── compute_pci.py          # Core PCI computation functions
│   └── analyze_combined.py     # Cross-dataset analysis pipeline
├── data/
│   ├── physionet_109_results.csv      # PhysioNet EEGBCI (N=109)
│   ├── openneuro_78_results.csv       # OpenNeuro ds003969 (N=78)
│   ├── lemon_207_results.csv          # LEMON dataset (N=207)
│   └── FINAL_COMBINED_394_results.csv # All datasets combined
└── figures/
    └── (generated figures)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic PCI Computation
```python
from scripts.compute_pci import compute_pci, compute_spectral_centroid

# Compute spectral centroids from PSD
theta_cent = compute_spectral_centroid(psd, freqs, 4, 8)
alpha_cent = compute_spectral_centroid(psd, freqs, 8, 13)

# Compute PCI
pci = compute_pci(alpha_cent, theta_cent)
print(f"PCI: {pci:.3f} ({'φ-organized' if pci > 0 else '2:1 organized'})")
```

### Run Combined Analysis
```bash
python scripts/analyze_combined.py
```

## Data Sources

1. **PhysioNet EEGBCI** (N=109): https://physionet.org/content/eegmmidb/1.0.0/
   - Schalk et al. (2004)
   - 64 channels, 160 Hz, eyes-closed resting state

2. **OpenNeuro ds003969** (N=78): https://openneuro.org/datasets/ds003969
   - Braboszcz et al. (2017)
   - 64 channels, 500 Hz, meditation baseline

3. **LEMON Mind-Brain-Body** (N=207): http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html
   - Babayan et al. (2019)
   - 62 channels, 2500 Hz, eyes-closed resting state

## Preprocessing

- Bandpass filter: 1-45 Hz (FIR)
- Bad channel interpolation (>3 SD variance threshold)
- Average reference (multi-channel)
- Epoch rejection: ±100 μV threshold
- PSD: Welch's method (4s windows, 50% overlap)

## Validation Procedures

1. **Null model simulation**: 100,000 synthetic datasets, effect exceeds null by >4 SD
2. **Per-dataset replication**: All 3 datasets show consistent effects (r = 0.55-0.67)
3. **Robust statistics**: Spearman ρ = 0.737 exceeds Pearson r = 0.576
4. **φ-specificity sweep**: Correlation peaks near φ = 1.618
5. **Epsilon sensitivity**: Stable across ε = 0.001 to 1.0
6. **Frontal theta validation**: r = 0.874 rules out volume conduction

## Citation

```bibtex
@article{ursachi2026golden,
  title={Golden Ratio Organization in Human EEG is Associated with 
         Theta-Alpha Frequency Convergence: A Multi-Dataset Validation Study},
  author={Ursachi, Andrei},
  journal={Frontiers in Human Neuroscience},
  year={2026},
  note={Manuscript ID: 1781338}
}
```

## Author

**Andrei Ursachi**  
Independent Researcher, Bucharest, Romania

## License

MIT License - see [LICENSE](LICENSE) for details.
