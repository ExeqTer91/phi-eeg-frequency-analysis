"""
Spectral analysis module.

Computes power spectral density and extracts peak frequencies
using both simple maximum and FOOOF parameterization methods.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import mne
from typing import Dict, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import BANDS, SPECTRAL, FOOOF_SETTINGS

# Try to import FOOOF
try:
    from fooof import FOOOF
    HAS_FOOOF = True
except ImportError:
    HAS_FOOOF = False


def compute_psd(
    raw: mne.io.Raw,
    window_sec: float = None,
    overlap: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    window_sec : float
        Window length in seconds
    overlap : float
        Overlap fraction (0-1)
        
    Returns
    -------
    freqs : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density (channels x frequencies)
    """
    if window_sec is None:
        window_sec = SPECTRAL['window_sec']
    if overlap is None:
        overlap = SPECTRAL['overlap']
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    nperseg = int(window_sec * sfreq)
    noverlap = int(nperseg * overlap)
    
    psd_list = []
    for ch_data in data:
        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=nperseg,
                                   noverlap=noverlap, window='hann')
        psd_list.append(psd)
    
    return freqs, np.array(psd_list)


def find_peak_simple(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float],
    smooth_sigma: float = 1.0
) -> Optional[float]:
    """
    Find peak frequency using smoothed maximum.
    
    Parameters
    ----------
    freqs : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density
    freq_range : tuple
        (low, high) frequency bounds
    smooth_sigma : float
        Gaussian smoothing sigma in bins
        
    Returns
    -------
    peak_freq : float or None
        Peak frequency in Hz
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask):
        return None
    
    freqs_band = freqs[mask]
    psd_band = psd[mask]
    
    # Average across channels if multi-channel
    if psd_band.ndim > 1:
        psd_band = np.mean(psd_band, axis=0)
    
    # Smooth
    psd_smooth = gaussian_filter1d(psd_band, sigma=smooth_sigma)
    
    # Find maximum
    peak_idx = np.argmax(psd_smooth)
    return freqs_band[peak_idx]


def find_peak_fooof(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float]
) -> Optional[float]:
    """
    Find peak frequency using FOOOF parameterization.
    
    Parameters
    ----------
    freqs : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density (averaged across channels)
    freq_range : tuple
        (low, high) frequency bounds for peak search
        
    Returns
    -------
    peak_freq : float or None
        Center frequency of detected peak
    """
    if not HAS_FOOOF:
        return None
    
    # Average across channels if needed
    if psd.ndim > 1:
        psd = np.mean(psd, axis=0)
    
    # Fit FOOOF model
    fm = FOOOF(**FOOOF_SETTINGS, verbose=False)
    
    try:
        fm.fit(freqs, psd, [1, 40])
        
        # Find peaks in target range
        peaks = fm.peak_params_
        if len(peaks) == 0:
            return None
        
        # Filter to freq_range
        in_range = [(p[0], p[1], p[2]) for p in peaks 
                    if freq_range[0] <= p[0] <= freq_range[1]]
        
        if not in_range:
            return None
        
        # Return highest power peak in range
        best_peak = max(in_range, key=lambda x: x[1])
        return best_peak[0]
        
    except Exception:
        return None


def extract_peak_frequencies(
    raw: mne.io.Raw,
    method: str = 'simple',
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, Optional[float]]:
    """
    Extract peak frequencies for theta and alpha bands.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    method : str
        'simple' or 'fooof'
    bands : dict
        Band definitions {name: (low, high)}
        
    Returns
    -------
    peaks : dict
        {band_name: peak_frequency}
    """
    if bands is None:
        bands = {'theta': BANDS['theta'], 'alpha': BANDS['alpha']}
    
    freqs, psd = compute_psd(raw)
    
    peaks = {}
    for band_name, freq_range in bands.items():
        if method == 'fooof' and HAS_FOOOF:
            peak = find_peak_fooof(freqs, psd, freq_range)
        else:
            peak = find_peak_simple(freqs, psd, freq_range)
        peaks[band_name] = peak
    
    return peaks


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float]
) -> float:
    """
    Compute integrated band power.
    
    Parameters
    ----------
    freqs : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density
    freq_range : tuple
        (low, high) frequency bounds
        
    Returns
    -------
    power : float
        Integrated power in band
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    
    if psd.ndim > 1:
        psd = np.mean(psd, axis=0)
    
    return np.trapz(psd[mask], freqs[mask])


def extract_power_ratio(
    raw: mne.io.Raw,
    numerator_band: str = 'alpha',
    denominator_band: str = 'theta'
) -> Optional[float]:
    """
    Compute power ratio between two bands.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    numerator_band : str
        Band name for numerator
    denominator_band : str
        Band name for denominator
        
    Returns
    -------
    ratio : float or None
        Power ratio
    """
    freqs, psd = compute_psd(raw)
    
    num_power = compute_band_power(freqs, psd, BANDS[numerator_band])
    den_power = compute_band_power(freqs, psd, BANDS[denominator_band])
    
    if den_power > 0:
        return num_power / den_power
    return None
