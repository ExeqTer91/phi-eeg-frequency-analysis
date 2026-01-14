"""
Visualization module.

Creates publication-quality figures for the frequency ratio analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PHI, FIGURES_DIR


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_ratio_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot distribution of frequency ratios by condition with reference lines.
    
    Parameters
    ----------
    df : DataFrame
        Results with 'condition' and 'freq_ratio' columns
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Violin plot
    conditions = df['condition'].unique()
    for i, cond in enumerate(conditions):
        data = df[df['condition'] == cond]['freq_ratio'].dropna()
        parts = ax.violinplot([data], positions=[i], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    
    # Reference lines
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='2:1 = 2.000')
    
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.upper() for c in conditions])
    ax.set_ylabel('α/θ Frequency Ratio', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title('State-Dependent Frequency Ratio Distribution', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_state_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot individual subject trajectories between conditions.
    
    Parameters
    ----------
    df : DataFrame
        Results with subject-level data
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pivot to get paired data
    pivot = df.pivot(index='subject', columns='condition', values='freq_ratio')
    
    if 'rest' in pivot.columns and 'task' in pivot.columns:
        # Plot individual trajectories
        for subj in pivot.index:
            rest_val = pivot.loc[subj, 'rest']
            task_val = pivot.loc[subj, 'task']
            if pd.notna(rest_val) and pd.notna(task_val):
                color = 'green' if task_val > rest_val else 'red'
                ax.plot([0, 1], [rest_val, task_val], 'o-', alpha=0.3, color=color)
        
        # Mean trajectory
        mean_rest = pivot['rest'].mean()
        mean_task = pivot['task'].mean()
        ax.plot([0, 1], [mean_rest, mean_task], 'ko-', linewidth=3, markersize=12,
                label=f'Mean: {mean_rest:.2f} → {mean_task:.2f}')
    
    # Reference lines
    ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='2:1 = 2.000')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['REST', 'TASK'])
    ax.set_ylabel('α/θ Frequency Ratio', fontsize=12)
    ax.set_title('Individual Subject Trajectories: REST → TASK', fontsize=14)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 5)
):
    """
    Plot histogram of subjects closer to φ vs 2:1 by condition.
    
    Parameters
    ----------
    df : DataFrame
        Results with 'closer_to' column
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for i, cond in enumerate(['rest', 'task']):
        ax = axes[i]
        cond_data = df[df['condition'] == cond]['closer_to'].dropna()
        
        counts = cond_data.value_counts()
        colors = ['gold' if x == 'phi' else 'red' for x in counts.index]
        
        bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Closer To', fontsize=12)
        ax.set_ylabel('Number of Subjects', fontsize=12)
        ax.set_title(f'{cond.upper()} Condition', fontsize=14)
    
    plt.suptitle('Model Comparison: φ vs 2:1', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_freq_vs_power_ratio(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 8)
):
    """
    Scatter plot of frequency ratio vs power ratio.
    
    Parameters
    ----------
    df : DataFrame
        Results with both ratio types
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for cond in df['condition'].unique():
        cond_data = df[df['condition'] == cond]
        ax.scatter(cond_data['freq_ratio'], cond_data['power_ratio'],
                   alpha=0.6, s=50, label=cond.upper())
    
    # Reference lines
    ax.axvline(x=PHI, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=2.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Frequency Ratio (f_α / f_θ)', fontsize=12)
    ax.set_ylabel('Power Ratio (P_α / P_θ)', fontsize=12)
    ax.set_title('Frequency Ratio vs Power Ratio\n(Different Measures)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_all_figures(df: pd.DataFrame, output_dir: Path = None):
    """
    Generate all publication figures.
    
    Parameters
    ----------
    df : DataFrame
        Analysis results
    output_dir : Path
        Directory to save figures
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating figures...")
    
    plot_ratio_distribution(df, output_dir / 'fig1_ratio_distribution.png')
    plot_state_comparison(df, output_dir / 'fig2_state_trajectories.png')
    plot_model_comparison(df, output_dir / 'fig3_model_comparison.png')
    plot_freq_vs_power_ratio(df, output_dir / 'fig4_freq_vs_power.png')
    
    print(f"Figures saved to {output_dir}")
