# Seed Equation: φ⁴ + φ⁻⁴ = 7

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: V73](https://img.shields.io/badge/Paper-V73_COMPLETE-blue.svg)](docs/V73_paper_summary.md)

## Overview

This repository contains reproducible code for the Seed Equation framework, which explores phenomenological connections between the golden ratio (φ), Lucas numbers, and nuclear physics.

**Core Identity:**
```
φ⁴ + φ⁻⁴ = 7 = L₄
```

**Central Prediction:** The next superheavy magic number is N = L₁₁ = **199** (not N = 184 as predicted by Standard Shell Model).

## Key Results

| Metric | Value | Status |
|--------|-------|--------|
| Magic-Lucas matches | 5/7 (71%) | Marginal (p = 0.17) |
| RCHB convergence | L₁₁ = 199 ≈ RCHB N = 198 | **Significant (p = 0.02)** |
| Planetary ratios | 3/28 (11%) | Not significant (p = 0.54) |

## Repository Structure

```
seed-equation/
├── nuclear/
│   ├── seed_equation_core.py      # Core calculations
│   ├── statistical_validation.py   # Monte Carlo simulations
│   └── README.md                   # Nuclear physics methodology
├── eeg/
│   └── README.md                   # EEG analysis documentation
├── docs/
│   └── V73_paper_summary.md        # Paper summary
└── README.md                       # This file
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/ExeqTer91/seed-equation.git
cd seed-equation

# Run core calculations
python nuclear/seed_equation_core.py

# Run statistical validation
python nuclear/statistical_validation.py
```

## Methodology

### Magic-Lucas Matching

Tolerance formula (symmetric):
```
|Magic - Lucas| / max(Magic, Lucas) ≤ 10%
```

**Matching pairs:** 20≈18, 28≈29, 50≈47, 82≈76, 126≈123

### RCHB Convergence (p = 0.02)

**Key finding**: N = 198 is a predicted magic number (Zhang 2005, Patra 2025, Saxena 2020). L₁₁ = 199 lies exactly 1 neutron away.

RCHB theory identifies candidate magic numbers in range N ≈ 150–250.
Probability that L₁₁ = 199 falls within ±1 of RCHB N = 198 by chance:
```
P = 2/100 = 0.02
```

## Falsifiable Prediction

| Model | Prediction | Basis |
|-------|------------|-------|
| Standard Shell Model | N = 184 | Non-relativistic |
| RCHB Theory | N = 198 | Relativistic QM |
| **Seed Equation** | **N = 199** | Lucas L₁₁ |

**Experimental test:** Synthesis of superheavy elements (Z > 118) and measurement of nuclear stability at N = 184 vs N ≈ 198-199.

## References

1. Zhang, W., et al. (2005). Magic numbers for superheavy nuclei in relativistic continuum Hartree-Bogoliubov theory. *Nuclear Physics A*, 753, 106-135.

2. Patra, S.K., et al. (2025). Shell structure and magic numbers in superheavy nuclei. *arXiv:2503.22260*.

3. Saxena, G., et al. (2020). Microscopic description of structural, surface, and decay properties of superheavy nuclei. *Nuclear Physics A*, 1003, 122011.

4. Coldea, R., et al. (2010). Quantum Criticality in an Ising Chain: Experimental Evidence for Emergent E8 Symmetry. *Science*, 327, 177-180.

## Author

**Andrei Ursachi**  
Independent Researcher, Bucharest, Romania  
ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:
```bibtex
@misc{ursachi2026seed,
  author = {Ursachi, Andrei},
  title = {Seed Equation: Phenomenological Convergence of Golden Ratio and Nuclear Magic Numbers},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ExeqTer91/seed-equation}
}
```
