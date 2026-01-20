#!/usr/bin/env python3
"""
φ-Compression Law Experiments

Full experimental protocol: 216 runs total
- 4 architectures × 3 scaling × 3 LRs × 6 seeds = 216 runs

Usage:
    python run_experiments.py --quick    # 10 epochs, subset
    python run_experiments.py --full     # 50 epochs, all runs
"""

import argparse
import json
import os
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm

from models import SimpleCNN, ResNet18Phi, ViTPhi
from scaling import get_layer_widths
from experiments import train_model, compute_metrics
from experiments.trainer import get_cifar10_loaders


CONFIGS = {
    'SimpleCNN': {
        'standard': [32, 64, 128],
        'lucas': [29, 47, 76],
        'fibonacci': [34, 55, 89],
    },
    'ResNet18': {
        'standard': [64, 128, 256, 512],
        'lucas': [47, 76, 123, 199],
        'fibonacci': [55, 89, 144, 233],
    },
    'ViT': {
        'standard': {'embed_dim': 192, 'num_heads': 3},
        'lucas': {'embed_dim': 123, 'num_heads': 3},
        'fibonacci': {'embed_dim': 144, 'num_heads': 3},
    }
}


def create_model(arch: str, scaling: str):
    """Create model with specified architecture and scaling."""
    config = CONFIGS[arch][scaling]
    
    if arch == 'SimpleCNN':
        return SimpleCNN(widths=config)
    elif arch == 'ResNet18':
        return ResNet18Phi(widths=config)
    elif arch == 'ViT':
        return ViTPhi(**config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def run_experiment(arch, scaling, lr, seed, epochs, train_loader, test_loader, device):
    """Run single experiment."""
    model = create_model(arch, scaling)
    results = train_model(
        model, train_loader, test_loader,
        epochs=epochs, lr=lr, seed=seed, device=device
    )
    return {
        'arch': arch,
        'scaling': scaling,
        'lr': lr,
        'seed': seed,
        'accuracy': results['accuracy'],
        'params': results['params']
    }


def main():
    parser = argparse.ArgumentParser(description='φ-Compression Law Experiments')
    parser.add_argument('--quick', action='store_true', help='Quick validation (10 epochs)')
    parser.add_argument('--full', action='store_true', help='Full experiments (50 epochs)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    epochs = 50 if args.full else (10 if args.quick else args.epochs)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"φ-Compression Law Experiments")
    print(f"Device: {device}, Epochs: {epochs}")
    print("=" * 60)
    
    train_loader, test_loader = get_cifar10_loaders()
    
    architectures = ['SimpleCNN', 'ResNet18', 'ViT']
    scalings = ['standard', 'lucas', 'fibonacci']
    lrs = [0.1, 0.05, 0.01] if args.full else [0.1]
    seeds = [42, 123, 456, 789, 1000, 2024] if args.full else [42]
    
    results = []
    total = len(architectures) * len(scalings) * len(lrs) * len(seeds)
    
    with tqdm(total=total, desc="Running experiments") as pbar:
        for arch in architectures:
            for scaling in scalings:
                for lr in lrs:
                    for seed in seeds:
                        result = run_experiment(
                            arch, scaling, lr, seed, epochs,
                            train_loader, test_loader, device
                        )
                        results.append(result)
                        pbar.update(1)
    
    df = pd.DataFrame(results)
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'results/experiments_{timestamp}.csv', index=False)
    
    print("\n" + "=" * 60)
    print("CONSERVATION LAW ANALYSIS")
    print("=" * 60)
    
    for arch in architectures:
        std_results = df[(df['arch'] == arch) & (df['scaling'] == 'standard')]
        lucas_results = df[(df['arch'] == arch) & (df['scaling'] == 'lucas')]
        
        if len(std_results) > 0 and len(lucas_results) > 0:
            baseline = {
                'accuracy': std_results['accuracy'].mean(),
                'params': std_results['params'].iloc[0]
            }
            phi = {
                'accuracy': lucas_results['accuracy'].mean(),
                'params': lucas_results['params'].iloc[0]
            }
            
            metrics = compute_metrics(baseline, phi)
            print(f"\n{arch}:")
            print(f"  Retention:  {metrics['retention']:.4f}")
            print(f"  Efficiency: {metrics['efficiency']:.4f}×")
            print(f"  Product:    {metrics['product']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to results/experiments_{timestamp}.csv")


if __name__ == '__main__':
    main()
