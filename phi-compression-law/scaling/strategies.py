"""
Scaling strategies for neural network layer widths.
Lucas numbers encode the conservation identity φⁿ × φ⁻ⁿ = 1.
"""

import math

PHI = (1 + math.sqrt(5)) / 2

LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364, 2207, 3571]
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]


def get_layer_widths(scaling_type: str, num_layers: int, base_width: int = 32) -> list:
    """
    Returns layer widths based on scaling strategy.
    
    Args:
        scaling_type: 'standard', 'lucas', 'fibonacci', 'pi', 'sqrt2'
        num_layers: Number of layers
        base_width: Starting width for standard/pi/sqrt2 scaling
    
    Returns:
        List of layer widths
    
    Examples:
        >>> get_layer_widths('standard', 3)
        [32, 64, 128]
        >>> get_layer_widths('lucas', 3)
        [29, 47, 76]
    """
    if scaling_type == 'standard':
        return [base_width * (2 ** i) for i in range(num_layers)]
    
    elif scaling_type == 'lucas':
        start_idx = 7
        return LUCAS[start_idx:start_idx + num_layers]
    
    elif scaling_type == 'fibonacci':
        start_idx = 9
        return FIBONACCI[start_idx:start_idx + num_layers]
    
    elif scaling_type == 'pi':
        widths = []
        w = base_width
        for _ in range(num_layers):
            widths.append(int(round(w)))
            w *= math.pi
        return widths
    
    elif scaling_type == 'sqrt2':
        widths = []
        w = base_width
        for _ in range(num_layers):
            widths.append(int(round(w)))
            w *= math.sqrt(2)
        return widths
    
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")


def compute_retention(phi_params: int, standard_params: int) -> float:
    """Compute parameter retention ratio."""
    return phi_params / standard_params


def compute_efficiency(accuracy: float, params: int) -> float:
    """Compute accuracy per million parameters."""
    return accuracy / (params / 1e6)


def compute_product(efficiency_gain: float, retention: float) -> float:
    """Compute the conservation law product."""
    return efficiency_gain * retention
