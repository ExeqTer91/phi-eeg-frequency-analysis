"""
Seed Equation Core Calculations
Matches V73 CLEAN paper exactly

Author: Andrei-Sebastian Ursachi
"""

import math

PHI = (1 + math.sqrt(5)) / 2

def lucas_number(n):
    """Calculate nth Lucas number using φⁿ + φ⁻ⁿ = Lₙ"""
    return round(PHI**n + (-PHI)**(-n))

def generate_lucas_sequence(max_n=15):
    """Generate Lucas sequence L₁ through L_max_n"""
    return {n: lucas_number(n) for n in range(1, max_n + 1)}

def verify_seed_identity():
    """Verify φ⁴ + φ⁻⁴ = 7 (the Seed Identity)"""
    result = PHI**4 + PHI**(-4)
    expected = 7
    error = abs(result - expected)
    return {
        'calculated': result,
        'expected': expected,
        'error': error,
        'exact': error < 1e-10
    }

MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

LUCAS_COMPARISON = {
    'L0': 2,   'L1': 1,   'L2': 3,   'L3': 4,   'L4': 7,
    'L5': 11,  'L6': 18,  'L7': 29,  'L8': 47,
    'L9': 76,  'L10': 123, 'L11': 199
}

def compare_magic_lucas(tolerance=0.10):
    """
    Compare magic numbers to Lucas numbers
    Returns matches within tolerance (default 10%)
    
    Tolerance formula: |Magic - Lucas| / max(Magic, Lucas) <= 10%
    
    Using max() as denominator creates a symmetric tolerance window that
    avoids bias toward smaller numbers. This yields 6/7 = 86% match rate.
    
    Matching pairs: 2≈2, 20≈18, 28≈29, 50≈47, 82≈76, 126≈123
    """
    lucas_values = list(LUCAS_COMPARISON.values())
    matches = []
    
    for magic in MAGIC_NUMBERS:
        for i, lucas in enumerate(lucas_values):
            if lucas == 0:
                continue
            rel_diff = abs(magic - lucas) / max(magic, lucas)
            if rel_diff <= tolerance:
                matches.append({
                    'magic': magic,
                    'lucas': lucas,
                    'lucas_index': i + 1,
                    'difference_pct': rel_diff * 100
                })
                break
    
    return {
        'matches': matches,
        'match_count': len(matches),
        'total_magic': len(MAGIC_NUMBERS),
        'match_rate': len(matches) / len(MAGIC_NUMBERS)
    }

def rchb_convergence():
    """
    Calculate convergence between Seed prediction and RCHB
    
    Seed Equation: N = L₁₁ = 199
    RCHB Theory: N = 198 (Zhang et al. 2005, Patra et al. 2025, Saxena 2020)
    
    RCHB candidates: 172, 184, 198, 228, 238
    """
    seed_prediction = 199
    rchb_prediction = 198
    standard_model = 184
    
    rchb_candidates = [172, 184, 198, 228, 238]
    
    difference = abs(seed_prediction - rchb_prediction)
    convergence_pct = difference / rchb_prediction * 100
    
    sm_deviation = abs(seed_prediction - standard_model) / seed_prediction * 100
    rchb_deviation = convergence_pct
    precision_ratio = round(sm_deviation / rchb_deviation, 1)
    
    return {
        'seed_prediction': seed_prediction,
        'rchb_prediction': rchb_prediction,
        'difference': difference,
        'convergence_pct': round(convergence_pct, 2),
        'standard_model': standard_model,
        'rchb_candidates': rchb_candidates,
        'sm_deviation_pct': round(sm_deviation, 1),
        'precision_ratio': precision_ratio
    }

if __name__ == "__main__":
    print("=" * 60)
    print("SEED EQUATION CORE VERIFICATION")
    print("=" * 60)
    
    print("\n1. SEED IDENTITY VERIFICATION")
    seed = verify_seed_identity()
    print(f"   φ⁴ + φ⁻⁴ = {seed['calculated']:.10f}")
    print(f"   Expected: {seed['expected']}")
    print(f"   EXACT: {seed['exact']} ✓" if seed['exact'] else "   ERROR!")
    
    print("\n2. LUCAS SEQUENCE")
    for name, val in LUCAS_COMPARISON.items():
        print(f"   {name} = {val}")
    
    print("\n3. MAGIC NUMBER vs LUCAS COMPARISON (10% tolerance)")
    comparison = compare_magic_lucas()
    print(f"   Matches: {comparison['match_count']}/{comparison['total_magic']}")
    print(f"   Match rate: {comparison['match_rate']*100:.1f}%")
    for m in comparison['matches']:
        print(f"   • Magic {m['magic']} ≈ L{m['lucas_index']}={m['lucas']} ({m['difference_pct']:.1f}% diff)")
    
    print("\n4. RCHB CONVERGENCE (THE KEY FINDING)")
    rchb = rchb_convergence()
    print(f"   Seed Equation (L₁₁): N = {rchb['seed_prediction']}")
    print(f"   RCHB Theory:         N = {rchb['rchb_prediction']}")
    print(f"   Standard Model:      N = {rchb['standard_model']}")
    print(f"   Convergence: {rchb['convergence_pct']}%")
    print(f"   RCHB candidates: {rchb['rchb_candidates']}")
    print(f"   Standard Model deviation: {rchb['sm_deviation_pct']}%")
    print(f"   RCHB deviation: {rchb['convergence_pct']}%")
    print(f"   Precision ratio: {rchb['precision_ratio']}× (RCHB is {rchb['precision_ratio']}× closer)")
    print("   → This is the paper's key finding!")
