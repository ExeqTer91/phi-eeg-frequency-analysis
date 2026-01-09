# The Seed Equation: Nuclear Physics Analysis

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This folder contains code for **"The Seed Equation" (V73)** - a phenomenological observation connecting the golden ratio (φ) and Lucas numbers to nuclear magic numbers.

**Central Prediction**: The next superheavy magic number is **N = 199** (Lucas L₁₁), contradicting the Standard Shell Model (N = 184) but converging within 0.5% of RCHB theory (N = 198).

## The Seed Identity

```
φ⁴ + φ⁻⁴ = 7 = L₄
```

Where φ = (1 + √5)/2 ≈ 1.618 (golden ratio) and L₄ is the 4th Lucas number.

## Key Finding: RCHB Convergence

| Model | Prediction | Basis |
|-------|------------|-------|
| Standard Shell Model | N = 184 | Non-relativistic |
| RCHB Theory | N = 198 | Relativistic QM |
| **Seed Equation** | **N = 199** | Lucas L₁₁ |

**Convergence**: 0.5% difference between Seed Equation and RCHB (p = 0.02, statistically significant)

## Files

| File | Description |
|------|-------------|
| `seed_equation_core.py` | Lucas sequences, magic number comparison, RCHB analysis |
| `statistical_validation.py` | Monte Carlo p-value calculations (Table 2 from paper) |

## Usage

```bash
# Run nuclear physics analysis
python seed_equation_core.py

# Run statistical validation (Monte Carlo, n=100,000)
python statistical_validation.py
```

## Key References

1. **Coldea et al. (2010)** - Golden ratio in quantum systems  
   [Science 327, 177-180](https://www.science.org/doi/10.1126/science.1180085)

2. **Zhang et al. (2005)** - RCHB magic number predictions (N=198)  
   [Nuclear Physics A, 753, 106-135](https://www.sciencedirect.com/science/article/abs/pii/S0375947405002423)

3. **Patra et al. (2025)** - Superheavy nuclei shell structure  
   arXiv:2503.22260

4. **Saxena et al. (2020)** - Superheavy structural properties  
   [Nuclear Physics A, 1003, 122011](https://www.sciencedirect.com/science/article/abs/pii/S0375947420301986)

## Methodology Note

**Tolerance formula**: `|Magic - Lucas| / max(Magic, Lucas) <= 10%`

Using max() as the denominator creates a symmetric tolerance window that avoids bias toward smaller numbers. This yields 5/7 = 71% match rate, with Magic 20 ≈ Lucas 18 at the 10% boundary.

## Statistical Summary (Table 2)

| Pattern | P-value | Significant? | Interpretation |
|---------|---------|--------------|----------------|
| Magic-Lucas | 0.14 | No | Marginal |
| **RCHB convergence** | **0.02** | **Yes** | **Significant** |
| Mass formula | N/A | No | Post-hoc |
| Planetary | 0.54 | No | Chance level |

*Only the RCHB convergence is statistically robust (p < 0.05)*

## Falsification Timeline

When superheavy nuclei with N > 184 are synthesized (20-30 years):
- **N = 184 only stable** → Seed Equation falsified
- **N ≈ 198-199 stable** → Seed Equation supported

## Author

Andrei-Sebastian Ursachi  
ORCID: 0009-0002-6114-5011

## License

MIT License
