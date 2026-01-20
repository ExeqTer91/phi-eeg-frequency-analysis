"""Visualization for φ-compression law experiments."""

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_frontier(results: list, save_path: str = None):
    """
    Plot accuracy vs parameters Pareto frontier.
    
    Args:
        results: List of dicts with 'name', 'accuracy', 'params', 'scaling'
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'standard': '#1f77b4', 'lucas': '#d62728', 'fibonacci': '#2ca02c'}
    markers = {'SimpleCNN': 'o', 'ResNet18': 's', 'ConvNeXt': '^', 'ViT': 'D'}
    
    for result in results:
        ax.scatter(
            result['params'] / 1e6,
            result['accuracy'],
            c=colors.get(result['scaling'], 'gray'),
            marker=markers.get(result['arch'], 'o'),
            s=100,
            label=f"{result['arch']} ({result['scaling']})"
        )
    
    ax.set_xlabel('Parameters (millions)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('φ-Compression Law: Accuracy vs Parameters', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_conservation_law(metrics_list: list, save_path: str = None):
    """
    Plot efficiency × retention = 1 conservation law.
    
    Args:
        metrics_list: List of dicts with 'arch', 'efficiency', 'retention', 'product'
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for m in metrics_list:
        ax.scatter(m['retention'], m['efficiency'], s=100, label=m['arch'])
    
    x = np.linspace(0.1, 1, 100)
    ax.plot(x, 1/x, 'k--', alpha=0.5, label='eff × ret = 1')
    
    ax.axhline(y=np.e, color='gray', linestyle=':', alpha=0.5, label=f'e ≈ {np.e:.3f}')
    ax.axvline(x=1/np.e, color='gray', linestyle=':', alpha=0.5, label=f'1/e ≈ {1/np.e:.3f}')
    
    ax.set_xlabel('Retention (φ_params / std_params)', fontsize=12)
    ax.set_ylabel('Efficiency Gain', fontsize=12)
    ax.set_title('φ-Compression Conservation Law', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
