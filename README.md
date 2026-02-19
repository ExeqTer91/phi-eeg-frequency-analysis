# Golden Ratio Organization in Human EEG: Analysis Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Frontiers](https://img.shields.io/badge/Paper-Frontiers%20in%20Human%20Neuroscience-blue.svg)](https://doi.org/10.3389/fnhum.2026.1781338)

## Overview

Analysis code for: **"Golden Ratio Organization in Human EEG is Associated with Theta-Alpha Frequency Convergence: A Multi-Dataset Validation Study"** (Ursachi, 2026), published in *Frontiers in Human Neuroscience*.

This repository contains all code to reproduce the Phi Coupling Index (PCI) analysis, validation procedures, and figures reported in the manuscript.

## Key Findings

- 80% of subjects (N=320) show phi-organized EEG spectral architecture
- PCI correlates with theta-alpha convergence (r = 0.54, p < 10^-25)
- Frontal theta validation yields r = 0.718, evidence against volume conduction
- Effect replicates across two independent datasets (PhysioNet EEGBCI, LEMON)

## Datasets

This study analyzed two publicly available EEG datasets:

- **PhysioNet EEGBCI** (N=109): https://physionet.org/content/eegmmidb/
- **LEMON Mind-Brain-Body** (N=211): https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html

Download both datasets before running the analysis. See instructions below.

## Installation

```bash
git clone https://github.com/ExeqTer91/eeg-phi-coupling.git
cd eeg-phi-coupling
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- MNE-Python
- NumPy, SciPy, Matplotlib
- FOOOF (for aperiodic correction)
- See `requirements.txt` for full list

## Usage

```bash
# Run full analysis pipeline
python analysis/pci_analysis.py

# Generate figures
python analysis/visualization.py
```

## Repository Structure

```
eeg-phi-coupling/
├── analysis/          # Core analysis scripts
│   ├── pci_analysis.py          # Main PCI computation
│   ├── spectral_centroids.py    # Theta/alpha centroid extraction
│   ├── preprocessing.py         # EEG preprocessing pipeline
│   ├── visualization.py         # Figure generation
│   ├── config.py                # Parameters and constants
│   ├── compute_pci.py           # Batch PCI computation
│   └── analyze_combined.py      # Multi-dataset analysis
├── results/           # Output data
│   ├── physionet_109_results.csv
│   ├── lemon_207_results.csv
│   ├── openneuro_78_results.csv
│   └── FINAL_COMBINED_394_results.csv
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
├── CITATION.cff       # Citation metadata
└── README.md          # This file
```

## Citation

```bibtex
@article{ursachi2026golden,
  author = {Ursachi, Andrei},
  title = {Golden Ratio Organization in Human EEG is Associated with Theta-Alpha Frequency Convergence: A Multi-Dataset Validation Study},
  journal = {Frontiers in Human Neuroscience},
  year = {2026},
  volume = {20},
  pages = {1781338},
  doi = {10.3389/fnhum.2026.1781338}
}
```

## Author

**Andrei Ursachi**
Independent Researcher, Bucharest, Romania
ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011)

## License

MIT License -- see [LICENSE](LICENSE) for details.
