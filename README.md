# EEG Peak Frequency Ratio Analysis

**State-Dependent Frequency Switching: φ vs 2:1 Dynamics in Neural Oscillations**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the analysis code for investigating state-dependent α/θ frequency ratio switching in human EEG. The core hypothesis: the brain dynamically shifts between **φ ≈ 1.618** (anti-resonance, decoupling) and **2:1** (harmonic coupling) frequency ratios based on cognitive demands.

**Associated manuscript**: "The Golden Ratio at Stability Boundaries: Evidence from Quantum Criticality, Neural Dynamics, and Black Hole Physics"

**Author**: Andrei Ursachi (ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011))

## Hypothesis

Based on Pletzer et al. (2010) and Rodriguez-Larios & Alaerts (2019):

| Cognitive State | Expected f_α/f_θ | Mechanism |
|-----------------|------------------|-----------|
| **Deep rest / Meditation** | → φ (1.618) | Anti-resonance, maximal decoupling |
| **Active task** | → 2:1 (2.0) | Harmonic coupling for integration |
| **Normal rest** | ~1.7-1.9 | Intermediate / neutral |

The golden ratio φ is the "most irrational" number (Hurwitz's theorem), making φ-ratios maximally resistant to phase-locking—a direct application of KAM theorem dynamics.

## Key Results

### PhysioNet EEGMMIDB (N = 51)

| Condition | Frequency Ratio (f_α/f_θ) | Interpretation |
|-----------|---------------------------|----------------|
| REST | 1.87 ± 0.47 | Intermediate |
| TASK | 2.00 ± 0.52 | → 2:1 harmonic |

**State-dependent switching confirmed**: TASK significantly shifts toward 2:1 coupling (p < 0.001, Cohen's d = 0.93)

### Meditation Analysis

| Dataset | Condition | f_α/f_θ | Distance from φ |
|---------|-----------|---------|-----------------|
| OpenNeuro ds003969 | Meditation | 1.65 ± 0.31 | 1.97% |

Meditation states converge toward φ, supporting the anti-resonance hypothesis.

## Installation

```bash
git clone https://github.com/yourusername/phi-eeg-frequency-analysis.git
cd phi-eeg-frequency-analysis
pip install -r requirements.txt
```

## Usage

### Quick Analysis

```python
from src.analysis import run_full_analysis

# Run complete pipeline
results = run_full_analysis(
    dataset='physionet',
    conditions=['rest', 'task'],
    output_dir='results/'
)
```

### Step-by-Step

```python
from src.preprocessing import load_and_preprocess
from src.spectral import extract_peak_frequencies
from src.analysis import compute_frequency_ratios

# Load data
eeg_data = load_and_preprocess('S001', condition='rest')

# Extract peaks
peaks = extract_peak_frequencies(eeg_data, method='fooof')

# Compute ratio
ratio = peaks['alpha'] / peaks['theta']
print(f"α/θ frequency ratio: {ratio:.3f}")
```

## Project Structure

```
phi-eeg-frequency-analysis/
├── README.md
├── requirements.txt
├── config.py                 # Band definitions, paths, parameters
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Load, filter, artifact rejection
│   ├── spectral.py          # PSD, peak extraction (simple + FOOOF)
│   ├── analysis.py          # Ratio computation, statistics
│   └── visualization.py     # All figures
├── results/
│   ├── figures/
│   └── tables/
└── data/                    # Downloaded datasets (gitignored)
```

## Datasets

### Primary: PhysioNet EEGMMIDB
- URL: https://physionet.org/content/eegmmidb/1.0.0/
- N = 109 subjects (51 after artifact rejection)
- 64 channels, 160 Hz sampling
- Conditions: REST (eyes open/closed), TASK (motor imagery)

### Cross-validation: OpenNeuro ds003969
- URL: https://openneuro.org/datasets/ds003969
- Meditation EEG dataset
- Expected to show ratios closer to φ

## Methods

### Preprocessing
1. Bandpass filter 1-45 Hz
2. Artifact rejection: ±100 μV threshold
3. Central/parietal channel selection (Cz, Pz, C3, C4, P3, P4)

### Spectral Analysis
- Welch's method: 2s windows, 50% overlap
- Peak extraction via FOOOF parameterization (Donoghue et al., 2020)
- Theta band: 4-8 Hz, Alpha band: 8-13 Hz

### Statistical Tests
- Paired t-test: REST vs TASK frequency ratios
- Effect size: Cohen's d
- Model comparison: % subjects closer to φ vs 2:1
- TOST equivalence test for meditation → φ

## Key References

1. **Pletzer et al. (2010)**. "When frequencies never synchronize: The golden mean and the resting EEG." *Brain Research* 1335:91-102.

2. **Rodriguez-Larios & Alaerts (2019)**. "Tracking transient changes in the neural frequency architecture." *Journal of Neuroscience* 39:6291-6298.

3. **Coldea et al. (2010)**. "Quantum criticality in an Ising chain: Experimental evidence for emergent E8 symmetry." *Science* 327:177-180.

4. **Donoghue et al. (2020)**. "Parameterizing neural power spectra into periodic and aperiodic components." *Nature Neuroscience* 23:1655-1665.

5. **Greene (1979)**. "A method for determining a stochastic transition." *Journal of Mathematical Physics* 20:1183-1201.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

```bibtex
@article{ursachi2026golden,
  title={The Golden Ratio at Stability Boundaries: Evidence from Quantum Criticality, Neural Dynamics, and Black Hole Physics},
  author={Ursachi, Andrei},
  journal={Chaos},
  year={2026}
}
```
