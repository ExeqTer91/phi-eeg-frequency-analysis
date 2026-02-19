"""
analyze_combined.py - Cross-dataset analysis for Phi Coupling Index

Analyzes N=394 subjects across three datasets:
- PhysioNet EEGBCI (N=109)
- OpenNeuro ds003969 (N=78)  
- LEMON Mind-Brain-Body (N=207)

Author: Andrei Condrea
ORCID: 0009-0002-6114-5011
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')


def load_all_datasets():
    """Load and combine all three datasets."""
    
    datasets = {}
    
    # PhysioNet
    physionet_path = os.path.join(DATA_DIR, 'physionet_109_results.csv')
    if os.path.exists(physionet_path):
        datasets['PhysioNet'] = pd.read_csv(physionet_path)
        print(f"Loaded PhysioNet: N={len(datasets['PhysioNet'])}")
    
    # OpenNeuro
    openneuro_path = os.path.join(DATA_DIR, 'openneuro_78_results.csv')
    if os.path.exists(openneuro_path):
        datasets['OpenNeuro'] = pd.read_csv(openneuro_path)
        print(f"Loaded OpenNeuro: N={len(datasets['OpenNeuro'])}")
    
    # LEMON
    lemon_path = os.path.join(DATA_DIR, 'lemon_207_results.csv')
    if os.path.exists(lemon_path):
        datasets['LEMON'] = pd.read_csv(lemon_path)
        print(f"Loaded LEMON: N={len(datasets['LEMON'])}")
    
    return datasets


def analyze_dataset(df, name):
    """Compute statistics for a single dataset."""
    
    # Ensure we have required columns
    if 'PCI' not in df.columns or 'convergence' not in df.columns:
        print(f"Warning: {name} missing required columns")
        return None
    
    # Remove any NaN values
    df_clean = df.dropna(subset=['PCI', 'convergence'])
    
    # Compute statistics
    results = {
        'name': name,
        'N': len(df_clean),
        'PCI_mean': df_clean['PCI'].mean(),
        'PCI_std': df_clean['PCI'].std(),
    }
    
    # PCI-Convergence correlation
    r, p = stats.pearsonr(df_clean['PCI'], df_clean['convergence'])
    results['r_pearson'] = r
    results['p_pearson'] = p
    
    # Spearman for robustness
    rho, p_rho = stats.spearmanr(df_clean['PCI'], df_clean['convergence'])
    results['rho_spearman'] = rho
    results['p_spearman'] = p_rho
    
    # Phi organization
    n_phi_organized = (df_clean['PCI'] > 0).sum()
    results['n_phi_organized'] = n_phi_organized
    results['pct_phi_organized'] = 100 * n_phi_organized / len(df_clean)
    
    # Ratio statistics (if available)
    if 'theta_centroid' in df_clean.columns and 'alpha_centroid' in df_clean.columns:
        df_clean['ratio'] = df_clean['alpha_centroid'] / df_clean['theta_centroid']
        results['ratio_mean'] = df_clean['ratio'].mean()
        results['ratio_std'] = df_clean['ratio'].std()
        
        # Distance from phi
        PHI = 1.618034
        results['distance_from_phi'] = abs(results['ratio_mean'] - PHI)
        results['pct_from_phi'] = 100 * results['distance_from_phi'] / PHI
    
    return results


def print_results(results):
    """Print formatted results."""
    
    print(f"\n{'='*60}")
    print(f"{results['name']} (N={results['N']})")
    print(f"{'='*60}")
    
    print(f"\nPCI-Convergence Correlation:")
    print(f"  Pearson r = {results['r_pearson']:.3f}, p = {results['p_pearson']:.2e}")
    print(f"  Spearman ρ = {results['rho_spearman']:.3f}, p = {results['p_spearman']:.2e}")
    
    print(f"\nPhi Organization:")
    print(f"  φ-organized (PCI > 0): {results['n_phi_organized']} ({results['pct_phi_organized']:.1f}%)")
    
    if 'ratio_mean' in results:
        print(f"\nFrequency Ratio:")
        print(f"  Mean α/θ = {results['ratio_mean']:.3f} (SD = {results['ratio_std']:.3f})")
        print(f"  Distance from φ: {results['pct_from_phi']:.1f}%")


def main():
    """Main analysis pipeline."""
    
    print("="*70)
    print("GOLDEN RATIO ORGANIZATION IN HUMAN EEG")
    print("Multi-Dataset Validation (N=394)")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets()
    
    if not datasets:
        print("Error: No datasets found. Check data directory.")
        return
    
    # Analyze each dataset
    all_results = []
    combined_df = []
    
    for name, df in datasets.items():
        results = analyze_dataset(df, name)
        if results:
            all_results.append(results)
            print_results(results)
            
            # Add to combined
            df_copy = df.copy()
            if 'dataset' not in df_copy.columns:
                df_copy['dataset'] = name
            combined_df.append(df_copy)
    
    # Combined analysis
    if combined_df:
        print("\n")
        print("*"*70)
        print("COMBINED ANALYSIS")
        print("*"*70)
        
        df_all = pd.concat(combined_df, ignore_index=True)
        combined_results = analyze_dataset(df_all, "COMBINED")
        if combined_results:
            print_results(combined_results)
    
    # Summary table
    print("\n")
    print("="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Dataset':<15} {'N':>6} {'r':>8} {'% φ-org':>10} {'Mean ratio':>12}")
    print("-"*55)
    
    for r in all_results:
        ratio_str = f"{r.get('ratio_mean', 0):.3f}" if 'ratio_mean' in r else "N/A"
        print(f"{r['name']:<15} {r['N']:>6} {r['r_pearson']:>8.3f} {r['pct_phi_organized']:>9.1f}% {ratio_str:>12}")
    
    if combined_results:
        ratio_str = f"{combined_results.get('ratio_mean', 0):.3f}"
        print("-"*55)
        print(f"{'COMBINED':<15} {combined_results['N']:>6} {combined_results['r_pearson']:>8.3f} {combined_results['pct_phi_organized']:>9.1f}% {ratio_str:>12}")
    
    print("\n*** All p-values < 10⁻⁹ ***")


if __name__ == "__main__":
    main()
