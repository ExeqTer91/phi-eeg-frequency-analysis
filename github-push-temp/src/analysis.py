"""
Statistical analysis module.

Computes frequency ratios, runs hypothesis tests, and generates
summary statistics for state-dependent switching analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PHI, TARGET_RATIOS, STATS, RESULTS_DIR, TABLES_DIR
from .preprocessing import load_and_preprocess, get_valid_subjects
from .spectral import extract_peak_frequencies, extract_power_ratio


def compute_frequency_ratios(
    subjects: List[int],
    conditions: List[str] = ['rest', 'task'],
    method: str = 'simple',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute α/θ frequency ratios for all subjects and conditions.
    
    Parameters
    ----------
    subjects : list of int
        Subject numbers to analyze
    conditions : list of str
        Conditions to analyze
    method : str
        Peak detection method ('simple' or 'fooof')
    verbose : bool
        Show progress bar
        
    Returns
    -------
    df : DataFrame
        Results with columns: subject, condition, freq_ratio, power_ratio,
        alpha_peak, theta_peak, closer_to
    """
    results = []
    
    iterator = tqdm(subjects, desc="Analyzing") if verbose else subjects
    
    for subj in iterator:
        for cond in conditions:
            raw = load_and_preprocess(subj, cond, verbose=False)
            if raw is None:
                continue
            
            # Extract peaks
            peaks = extract_peak_frequencies(raw, method=method)
            alpha_peak = peaks.get('alpha')
            theta_peak = peaks.get('theta')
            
            # Compute frequency ratio
            if alpha_peak and theta_peak and theta_peak > 0:
                freq_ratio = alpha_peak / theta_peak
            else:
                freq_ratio = None
            
            # Compute power ratio
            power_ratio = extract_power_ratio(raw)
            
            # Determine which target is closer
            closer_to = None
            if freq_ratio is not None:
                dist_phi = abs(freq_ratio - PHI)
                dist_2 = abs(freq_ratio - 2.0)
                closer_to = 'phi' if dist_phi < dist_2 else '2:1'
            
            results.append({
                'subject': subj,
                'condition': cond,
                'freq_ratio': freq_ratio,
                'power_ratio': power_ratio,
                'alpha_peak': alpha_peak,
                'theta_peak': theta_peak,
                'dist_to_phi': abs(freq_ratio - PHI) if freq_ratio else None,
                'dist_to_2': abs(freq_ratio - 2.0) if freq_ratio else None,
                'closer_to': closer_to
            })
    
    return pd.DataFrame(results)


def run_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    Run all statistical tests for the frequency ratio analysis.
    
    Parameters
    ----------
    df : DataFrame
        Results from compute_frequency_ratios
        
    Returns
    -------
    results : dict
        Dictionary of test results
    """
    results = {}
    
    # Get paired data (subjects with both conditions)
    rest = df[df['condition'] == 'rest'].set_index('subject')
    task = df[df['condition'] == 'task'].set_index('subject')
    common = rest.index.intersection(task.index)
    
    rest_ratios = rest.loc[common, 'freq_ratio'].dropna()
    task_ratios = task.loc[common, 'freq_ratio'].dropna()
    
    # Find common subjects with valid ratios in both
    common_valid = rest_ratios.index.intersection(task_ratios.index)
    rest_paired = rest_ratios.loc[common_valid]
    task_paired = task_ratios.loc[common_valid]
    
    results['n_paired'] = len(common_valid)
    
    # 1. Paired t-test: REST vs TASK
    if len(common_valid) > 2:
        t_stat, p_value = stats.ttest_rel(task_paired, rest_paired)
        
        # Cohen's d
        diff = task_paired - rest_paired
        cohens_d = diff.mean() / diff.std()
        
        results['paired_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_rest': rest_paired.mean(),
            'mean_task': task_paired.mean(),
            'std_rest': rest_paired.std(),
            'std_task': task_paired.std()
        }
    
    # 2. Model comparison: closer to φ vs 2:1 by condition
    for cond in ['rest', 'task']:
        cond_data = df[(df['condition'] == cond) & (df['closer_to'].notna())]
        n_phi = (cond_data['closer_to'] == 'phi').sum()
        n_2 = (cond_data['closer_to'] == '2:1').sum()
        n_total = len(cond_data)
        
        # Binomial test: is proportion significantly different from 50%?
        if n_total > 0:
            binom_p = stats.binom_test(n_phi, n_total, 0.5, alternative='two-sided')
            results[f'{cond}_model_comparison'] = {
                'n_closer_phi': n_phi,
                'n_closer_2': n_2,
                'pct_phi': 100 * n_phi / n_total,
                'pct_2': 100 * n_2 / n_total,
                'binomial_p': binom_p
            }
    
    # 3. Distance comparison: are ratios significantly closer to one target?
    for cond in ['rest', 'task']:
        cond_data = df[(df['condition'] == cond)]
        dist_phi = cond_data['dist_to_phi'].dropna()
        dist_2 = cond_data['dist_to_2'].dropna()
        
        if len(dist_phi) > 2:
            t_stat, p_value = stats.ttest_rel(dist_phi, dist_2)
            results[f'{cond}_distance_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_dist_phi': dist_phi.mean(),
                'mean_dist_2': dist_2.mean()
            }
    
    return results


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics table.
    
    Parameters
    ----------
    df : DataFrame
        Results from compute_frequency_ratios
        
    Returns
    -------
    summary : DataFrame
        Summary statistics by condition
    """
    summary = []
    
    for cond in df['condition'].unique():
        cond_data = df[df['condition'] == cond]
        ratios = cond_data['freq_ratio'].dropna()
        
        if len(ratios) > 0:
            summary.append({
                'Condition': cond.upper(),
                'N': len(ratios),
                'Mean': ratios.mean(),
                'SD': ratios.std(),
                'Median': ratios.median(),
                '95% CI Lower': ratios.mean() - 1.96 * ratios.std() / np.sqrt(len(ratios)),
                '95% CI Upper': ratios.mean() + 1.96 * ratios.std() / np.sqrt(len(ratios)),
                'Deviation from φ': abs(ratios.mean() - PHI),
                'Deviation from 2:1': abs(ratios.mean() - 2.0),
                '% Closer to φ': 100 * (cond_data['closer_to'] == 'phi').sum() / len(ratios)
            })
    
    return pd.DataFrame(summary)


def run_full_analysis(
    dataset: str = 'physionet',
    conditions: List[str] = ['rest', 'task'],
    max_subjects: int = None,
    output_dir: Path = None,
    verbose: bool = True
) -> Dict:
    """
    Run complete analysis pipeline.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('physionet')
    conditions : list of str
        Conditions to analyze
    max_subjects : int
        Maximum number of subjects
    output_dir : Path
        Output directory for results
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Complete analysis results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir = Path(output_dir)
    
    # Get valid subjects
    if verbose:
        print("Step 1: Identifying valid subjects...")
    subjects = get_valid_subjects(conditions[0], max_subjects, verbose=verbose)
    
    # Compute ratios
    if verbose:
        print("\nStep 2: Computing frequency ratios...")
    df = compute_frequency_ratios(subjects, conditions, verbose=verbose)
    
    # Run statistics
    if verbose:
        print("\nStep 3: Running statistical tests...")
    stats_results = run_statistical_tests(df)
    
    # Generate summary
    summary = generate_summary_table(df)
    
    # Save results
    df.to_csv(output_dir / 'tables' / 'subject_level_results.csv', index=False)
    summary.to_csv(output_dir / 'tables' / 'summary_statistics.csv', index=False)
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(summary.to_string(index=False))
        
        if 'paired_ttest' in stats_results:
            pt = stats_results['paired_ttest']
            print(f"\nPaired t-test (TASK vs REST):")
            print(f"  t = {pt['t_statistic']:.3f}, p = {pt['p_value']:.4f}")
            print(f"  Cohen's d = {pt['cohens_d']:.3f}")
    
    return {
        'data': df,
        'summary': summary,
        'statistics': stats_results
    }
