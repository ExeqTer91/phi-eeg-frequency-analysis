"""
Statistical Validation for Seed Equation V73
Monte Carlo simulations and probability calculations

CRITICAL: These calculations must EXACTLY match the paper.
Do NOT adjust code to match desired results - that's fraud.

Author: Andrei-Sebastian Ursachi
"""

import random
import math

PHI = (1 + math.sqrt(5)) / 2
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]
LUCAS_NUMBERS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]

def monte_carlo_magic_lucas(n_simulations=100000, tolerance=0.10, seed=42):
    """
    Monte Carlo simulation for Magic-Lucas correspondence
    
    NULL HYPOTHESIS: Magic numbers are random integers in [2, 126]
    TEST: How often do 7 random numbers match Lucas numbers within tolerance?
    
    Methodology (must match paper):
    1. Generate 7 random integers in range [2, 126]
    2. Count how many match ANY Lucas number within tolerance
    3. Tolerance formula: |value - lucas| / max(value, lucas) <= 10%
    4. Repeat n_simulations times
    5. p-value = fraction with >= 5 matches (observed = 5/7)
    """
    random.seed(seed)
    
    observed_matches = 5
    
    count_as_good_or_better = 0
    
    for _ in range(n_simulations):
        random_magics = [random.randint(2, 126) for _ in range(7)]
        
        matches = 0
        for rm in random_magics:
            for lucas in LUCAS_NUMBERS:
                if lucas == 0:
                    continue
                if abs(rm - lucas) / max(rm, lucas) <= tolerance:
                    matches += 1
                    break
        
        if matches >= observed_matches:
            count_as_good_or_better += 1
    
    p_value = count_as_good_or_better / n_simulations
    
    return {
        'n_simulations': n_simulations,
        'observed_matches': observed_matches,
        'count_as_good_or_better': count_as_good_or_better,
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'methodology': 'Random integers [2,126], symmetric 10% tolerance (÷max), >= 5 matches'
    }

def rchb_convergence_probability():
    """
    Calculate probability of RCHB convergence by chance
    
    METHODOLOGY:
    - Assume random prediction in range [150, 250] (superheavy region)
    - RCHB predicts N = 198
    - Seed predicts N = 199
    - What's the probability of landing within 0.5% of 198?
    
    Window: 198 ± 0.5% = 198 ± 0.99 ≈ [197, 199]
    Width: ~2 integers out of 100 possible
    p = 2/100 = 0.02
    """
    rchb_value = 198
    tolerance_pct = 0.5 / 100
    
    range_min = 150
    range_max = 250
    range_width = range_max - range_min
    
    window = rchb_value * tolerance_pct
    window_width = 2 * window
    
    p_value = window_width / range_width
    
    return {
        'rchb_prediction': rchb_value,
        'seed_prediction': 199,
        'tolerance': '0.5%',
        'range': f'[{range_min}, {range_max}]',
        'window_width': round(window_width, 2),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'methodology': f'Random prediction in [{range_min},{range_max}], 0.5% tolerance'
    }

def planetary_binomial_test():
    """
    Binomial test for planetary orbital ratios
    
    METHODOLOGY:
    - 8 planets = 28 pairs
    - Observed: 3 pairs within 5% of φ or φ²
    - Baseline expectation: ~10% by chance
    - p-value from binomial distribution
    """
    from math import comb
    
    n = 28
    k = 3
    p_baseline = 0.10
    
    p_value = 0
    for i in range(k, n + 1):
        p_value += comb(n, i) * (p_baseline ** i) * ((1 - p_baseline) ** (n - i))
    
    return {
        'total_pairs': n,
        'observed_matches': k,
        'baseline_probability': p_baseline,
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'methodology': 'Binomial test, 10% baseline, >= 3 matches'
    }

def print_summary_table():
    """Print Table 2 from paper"""
    
    magic_lucas = monte_carlo_magic_lucas()
    rchb = rchb_convergence_probability()
    planetary = planetary_binomial_test()
    
    print("\n" + "=" * 70)
    print("TABLE 2: STATISTICAL ASSESSMENT OF PATTERN CLAIMS")
    print("=" * 70)
    print(f"\n{'Pattern':<25} {'P-value':<12} {'Significant?':<15} {'Interpretation':<20}")
    print("-" * 70)
    print(f"{'Magic-Lucas':<25} {magic_lucas['p_value']:<12.2f} {'No':<15} {'Marginal':<20}")
    print(f"{'RCHB convergence':<25} {rchb['p_value']:<12.2f} {'Yes*':<15} {'Significant':<20}")
    print(f"{'Mass formula':<25} {'N/A':<12} {'No':<15} {'Post-hoc':<20}")
    print(f"{'Planetary':<25} {planetary['p_value']:<12.2f} {'No':<15} {'Chance level':<20}")
    print("-" * 70)
    print("\n* Only statistically robust finding (p < 0.05)")
    print("\nMethodologies:")
    print(f"  Magic-Lucas: {magic_lucas['methodology']}")
    print(f"  RCHB: {rchb['methodology']}")
    print(f"  Planetary: {planetary['methodology']}")

if __name__ == "__main__":
    print("=" * 60)
    print("STATISTICAL VALIDATION - V73 PAPER")
    print("=" * 60)
    
    print("\n1. MAGIC-LUCAS MONTE CARLO TEST")
    print("-" * 40)
    result = monte_carlo_magic_lucas()
    print(f"   Simulations: {result['n_simulations']:,}")
    print(f"   Observed matches: {result['observed_matches']}/7")
    print(f"   As good or better: {result['count_as_good_or_better']:,}")
    print(f"   P-value: {result['p_value']}")
    print(f"   Significant: {result['significant']}")
    
    print("\n2. RCHB CONVERGENCE PROBABILITY")
    print("-" * 40)
    result = rchb_convergence_probability()
    print(f"   Seed prediction: N = {result['seed_prediction']}")
    print(f"   RCHB prediction: N = {result['rchb_prediction']}")
    print(f"   Tolerance: {result['tolerance']}")
    print(f"   Search range: {result['range']}")
    print(f"   P-value: {result['p_value']}")
    print(f"   Significant: {result['significant']}")
    
    print("\n3. PLANETARY BINOMIAL TEST")
    print("-" * 40)
    result = planetary_binomial_test()
    print(f"   Total pairs: {result['total_pairs']}")
    print(f"   Observed matches: {result['observed_matches']}")
    print(f"   P-value: {result['p_value']}")
    print(f"   Significant: {result['significant']}")
    
    print_summary_table()
