# φ-Compression Law: Golden Ratio Neural Network Scaling

This repository contains the experimental code for validating the **φ-Compression Law**:

> When neural network layer widths follow golden ratio (φ ≈ 1.618) scaling via Lucas numbers, 
> the product of efficiency gain and parameter retention equals unity:
>
> **efficiency × retention = 1.0004 ≈ 1**

## Key Findings

- **2.81× efficiency gain** across CNN architectures (SimpleCNN, ResNet-18, ConvNeXt-Tiny)
- **64.4% parameter reduction** with <0.5% accuracy loss
- Efficiency approaches **Euler's number (e ≈ 2.718)**
- Retention approaches **1/e ≈ 0.368**
- **Architecture-specific**: CNNs benefit, standard transformers do NOT
- **φ-Transformer**: Introducing hierarchy to transformers recovers the conservation law

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/phi-compression-law.git
cd phi-compression-law

# Install dependencies
pip install -r requirements.txt

# Run quick validation (10 epochs, subset of experiments)
python run_experiments.py --quick

# Run full experiments (50 epochs, all 216 runs)
python run_experiments.py --full
```

## The Conservation Law

The mathematical foundation connects:

- **Lucas identity**: φⁿ + φ⁻ⁿ = Lₙ
- **Conservation property**: φⁿ × φ⁻ⁿ = 1
- **Empirical result**: efficiency × retention = 1

| Architecture | Efficiency Gain | Param Reduction | Product |
|--------------|-----------------|-----------------|---------|
| SimpleCNN    | 2.80×           | 64.4%           | 0.997   |
| ResNet-18    | 2.88×           | 65.3%           | 1.000   |
| ConvNeXt-Tiny| 2.75×           | 63.5%           | 1.004   |
| ViT-Tiny     | 1.00×           | 0%              | N/A     |
| **Mean ± SD**| **2.81 ± 0.07×**| **64.4 ± 0.9%** | **1.0004** |

## Layer Width Configurations

### Standard (Power of 2)
```
32 → 64 → 128 → 256
```

### Lucas (φ-scaled)
```
29 → 47 → 76 → 123
```

### Why Lucas, not Fibonacci?

- **Fibonacci**: Fₙ = (φⁿ − ψⁿ)/√5 → encodes *difference*
- **Lucas**: Lₙ = φⁿ + ψⁿ → encodes *sum* and directly reflects φⁿ × φ⁻ⁿ = 1

Fibonacci achieves efficiency × retention ≈ 1.02-1.05  
Lucas achieves efficiency × retention ≈ 1.0004 (exact conservation)

## Why Transformers Don't Benefit

Standard transformers use **uniform** layer dimensions:
```
768 → 768 → 768 → 768 → 768 → 768
```

No hierarchical "slack" exists for φ-scaling to optimize. This is the **uniform exhaustion hypothesis**.

The **φ-Transformer** introduces hierarchy:
```
embed_dims: 123 → 199 → 322 → 521
```

This recovers the conservation law (efficiency ≈ 2.76× ≈ e).

## Repository Structure

```
phi-compression-law/
├── README.md
├── requirements.txt
├── config.yaml
├── run_experiments.py          # Main entry point
├── models/
│   ├── simple_cnn.py           # SimpleCNN with configurable widths
│   ├── resnet18_phi.py         # ResNet-18 with φ-scaling
│   └── vit_phi.py              # Vision Transformer (control)
├── scaling/
│   └── strategies.py           # Lucas, Fibonacci, Standard, Pi, Sqrt2 scaling
├── experiments/
│   ├── trainer.py              # Training loop with metrics
│   └── evaluator.py            # Efficiency/retention calculations
├── results/
│   └── .gitkeep
└── analysis/
    └── visualize.py            # Pareto frontiers, conservation law plots
```

## Citation

```bibtex
@misc{ursachi2026phi,
  author = {Ursachi, Andrei},
  title = {The φ-Compression Law: Golden Ratio Scaling Yields e-Optimal Efficiency in Hierarchical Neural Networks},
  year = {2026},
  url = {https://github.com/yourusername/phi-compression-law}
}
```

## License

MIT License

## Contact

Andrei Ursachi - Independent Researcher, Bucharest, Romania  
ORCID: 0009-0002-6114-5011
