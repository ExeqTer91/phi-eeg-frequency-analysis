"""Efficiency and retention calculations for φ-compression law."""

import math


E = math.e
PHI = (1 + math.sqrt(5)) / 2


def compute_metrics(baseline_results: dict, phi_results: dict) -> dict:
    """
    Compute the conservation law metrics.
    
    efficiency = accuracy_per_param_phi / accuracy_per_param_baseline
    retention = phi_params / baseline_params
    product = efficiency × retention  (should equal ≈ 1.0 for Lucas scaling)
    
    Returns:
        dict with efficiency, retention, product, param_reduction, accuracy_delta
    """
    baseline_acc = baseline_results['accuracy']
    baseline_params = baseline_results['params']
    phi_acc = phi_results['accuracy']
    phi_params = phi_results['params']
    
    baseline_efficiency = baseline_acc / (baseline_params / 1e6)
    phi_efficiency = phi_acc / (phi_params / 1e6)
    
    efficiency = phi_efficiency / baseline_efficiency
    retention = phi_params / baseline_params
    product = efficiency * retention
    
    param_reduction = 1 - retention
    accuracy_delta = phi_acc - baseline_acc
    
    return {
        'efficiency': efficiency,
        'retention': retention,
        'product': product,
        'param_reduction': param_reduction,
        'accuracy_delta': accuracy_delta,
        'e_deviation': abs(efficiency - E) / E,
        'inv_e_deviation': abs(retention - 1/E) / (1/E)
    }


def format_results(metrics: dict) -> str:
    """Format metrics for display."""
    return f"""
φ-Compression Law Metrics:
  Efficiency Gain: {metrics['efficiency']:.4f}× (target: e ≈ 2.7183)
  Retention:       {metrics['retention']:.4f} (target: 1/e ≈ 0.3679)
  Product:         {metrics['product']:.4f} (target: 1.0000)
  
  Param Reduction: {metrics['param_reduction']*100:.1f}%
  Accuracy Delta:  {metrics['accuracy_delta']:+.2f}%
  
  Deviation from e:   {metrics['e_deviation']*100:.2f}%
  Deviation from 1/e: {metrics['inv_e_deviation']*100:.2f}%
"""
