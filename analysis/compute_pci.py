"""
compute_pci.py - Core functions for Phi Coupling Index analysis

Golden Ratio Organization in Human EEG
Author: Andrei Condrea
ORCID: 0009-0002-6114-5011
"""

import numpy as np
from scipy import signal

# Constants
PHI = 1.618034  # Golden ratio
EPSILON = 0.1   # Regularization parameter
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)


def compute_psd_welch(data, sfreq, fmin=1, fmax=45):
    """
    Compute Power Spectral Density using Welch's method.
    
    Parameters
    ----------
    data : array
        EEG time series (samples,)
    sfreq : float
        Sampling frequency in Hz
    fmin, fmax : float
        Frequency range of interest
        
    Returns
    -------
    freqs : array
        Frequency values
    psd : array
        Power spectral density values
    """
    nperseg = min(int(4 * sfreq), len(data))  # 4-second windows
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, noverlap=nperseg//2)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def compute_spectral_centroid(psd, freqs, fmin, fmax):
    """
    Compute spectral centroid (center of mass) within frequency band.
    
    Parameters
    ----------
    psd : array
        Power spectral density
    freqs : array
        Corresponding frequencies
    fmin, fmax : float
        Band limits in Hz
        
    Returns
    -------
    float
        Centroid frequency in Hz
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[mask]
    psd_band = psd[mask]
    
    if psd_band.sum() == 0:
        return np.nan
    
    return np.sum(freqs_band * psd_band) / np.sum(psd_band)


def compute_peak_frequency(psd, freqs, fmin, fmax):
    """
    Find peak frequency within frequency band.
    
    Parameters
    ----------
    psd : array
        Power spectral density
    freqs : array
        Corresponding frequencies
    fmin, fmax : float
        Band limits in Hz
        
    Returns
    -------
    float
        Peak frequency in Hz
    """
    from scipy.ndimage import uniform_filter1d
    
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[mask]
    psd_band = psd[mask]
    
    if len(psd_band) == 0:
        return np.nan
    
    # Smooth to avoid noise peaks
    psd_smooth = uniform_filter1d(psd_band, size=3)
    return freqs_band[np.argmax(psd_smooth)]


def compute_pci(f_alpha, f_theta, epsilon=EPSILON):
    """
    Compute Phi Coupling Index.
    
    PCI = log((|R - 2| + ε) / (|R - φ| + ε))
    
    Where R = f_alpha / f_theta
    
    Parameters
    ----------
    f_alpha : float
        Alpha frequency (centroid or peak) in Hz
    f_theta : float
        Theta frequency (centroid or peak) in Hz
    epsilon : float
        Regularization parameter (default: 0.1)
        
    Returns
    -------
    float
        PCI value
        - PCI > 0: closer to φ (1.618) than to 2:1
        - PCI < 0: closer to 2:1 than to φ
    """
    if np.isnan(f_alpha) or np.isnan(f_theta) or f_theta == 0:
        return np.nan
    
    ratio = f_alpha / f_theta
    
    distance_to_harmonic = np.abs(ratio - 2.0)
    distance_to_phi = np.abs(ratio - PHI)
    
    return np.log((distance_to_harmonic + epsilon) / (distance_to_phi + epsilon))


def compute_convergence(f_alpha, f_theta):
    """
    Compute theta-alpha convergence metric.
    
    Convergence = 1 / |f_alpha - f_theta|
    
    Parameters
    ----------
    f_alpha : float
        Alpha frequency in Hz
    f_theta : float
        Theta frequency in Hz
        
    Returns
    -------
    float
        Convergence value (higher = frequencies closer together)
    """
    diff = np.abs(f_alpha - f_theta)
    if diff == 0:
        return np.nan
    return 1 / diff


def analyze_subject(data, sfreq, posterior_channels=None, frontal_channels=None):
    """
    Complete analysis pipeline for a single subject.
    
    Parameters
    ----------
    data : array
        EEG data (channels x samples)
    sfreq : float
        Sampling frequency
    posterior_channels : list
        Indices of posterior channels for averaging
    frontal_channels : list, optional
        Indices of frontal channels for theta validation
        
    Returns
    -------
    dict
        Dictionary with all computed metrics
    """
    results = {}
    
    # Average posterior channels
    if posterior_channels is not None:
        posterior_data = data[posterior_channels].mean(axis=0)
    else:
        posterior_data = data.mean(axis=0)
    
    # Compute PSD
    freqs, psd = compute_psd_welch(posterior_data, sfreq)
    
    # Compute spectral metrics
    results['theta_centroid'] = compute_spectral_centroid(psd, freqs, *THETA_BAND)
    results['alpha_centroid'] = compute_spectral_centroid(psd, freqs, *ALPHA_BAND)
    results['theta_peak'] = compute_peak_frequency(psd, freqs, *THETA_BAND)
    results['alpha_peak'] = compute_peak_frequency(psd, freqs, *ALPHA_BAND)
    
    # Compute derived metrics
    results['ratio'] = results['alpha_centroid'] / results['theta_centroid']
    results['PCI'] = compute_pci(results['alpha_centroid'], results['theta_centroid'])
    results['convergence'] = compute_convergence(results['alpha_centroid'], results['theta_centroid'])
    
    # Frontal theta analysis (if channels provided)
    if frontal_channels is not None and len(frontal_channels) >= 2:
        frontal_data = data[frontal_channels].mean(axis=0)
        freqs_f, psd_f = compute_psd_welch(frontal_data, sfreq)
        results['theta_frontal'] = compute_spectral_centroid(psd_f, freqs_f, *THETA_BAND)
        
        # Frontal-based PCI (frontal theta, posterior alpha)
        results['PCI_frontal'] = compute_pci(results['alpha_centroid'], results['theta_frontal'])
        results['convergence_frontal'] = compute_convergence(results['alpha_centroid'], results['theta_frontal'])
    
    return results


def epsilon_sensitivity(f_alpha, f_theta, epsilons=None):
    """
    Test PCI stability across epsilon values.
    
    Parameters
    ----------
    f_alpha : float
        Alpha frequency
    f_theta : float  
        Theta frequency
    epsilons : list, optional
        Epsilon values to test (default: [0.001, 0.01, 0.1, 0.5, 1.0])
        
    Returns
    -------
    dict
        Epsilon -> PCI mapping
    """
    if epsilons is None:
        epsilons = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    return {eps: compute_pci(f_alpha, f_theta, epsilon=eps) for eps in epsilons}


if __name__ == "__main__":
    # Example usage
    print("Phi Coupling Index Analysis")
    print("="*50)
    print(f"Golden ratio (φ): {PHI}")
    print(f"Theta band: {THETA_BAND} Hz")
    print(f"Alpha band: {ALPHA_BAND} Hz")
    print()
    
    # Example calculation
    f_theta = 6.0  # Hz
    f_alpha = 10.0  # Hz
    
    ratio = f_alpha / f_theta
    pci = compute_pci(f_alpha, f_theta)
    conv = compute_convergence(f_alpha, f_theta)
    
    print(f"Example: θ={f_theta} Hz, α={f_alpha} Hz")
    print(f"  Ratio: {ratio:.3f}")
    print(f"  PCI: {pci:.3f} ({'φ-organized' if pci > 0 else '2:1 organized'})")
    print(f"  Convergence: {conv:.3f}")
