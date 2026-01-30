"""
EEG preprocessing module.

Handles loading, filtering, and artifact rejection for PhysioNet EEGMMIDB
and OpenNeuro datasets.
"""

import numpy as np
import mne
from mne.datasets import eegbci
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PREPROC, PHYSIONET, DATA_DIR


def load_physionet_subject(
    subject: int,
    runs: List[int],
    preload: bool = True
) -> mne.io.Raw:
    """
    Load PhysioNet EEGMMIDB data for a single subject.
    
    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int
        Run numbers to load (e.g., [1, 2] for rest)
    preload : bool
        Whether to preload data into memory
        
    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw EEG data
    """
    raw_files = eegbci.load_data(subject, runs, path=str(DATA_DIR))
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=preload) 
                                 for f in raw_files])
    eegbci.standardize(raw)
    raw.set_montage('standard_1005', on_missing='ignore')
    return raw


def preprocess_raw(
    raw: mne.io.Raw,
    bandpass: Tuple[float, float] = None,
    channels: List[str] = None
) -> mne.io.Raw:
    """
    Apply preprocessing to raw EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    bandpass : tuple of float
        (low, high) cutoff frequencies in Hz
    channels : list of str
        Channel names to select
        
    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    if bandpass is None:
        bandpass = PREPROC['bandpass']
    if channels is None:
        channels = PREPROC['channels']
    
    raw = raw.copy()
    
    # Select channels that exist
    available = [ch for ch in channels if ch in raw.ch_names]
    if available:
        raw.pick_channels(available)
    
    # Bandpass filter
    raw.filter(bandpass[0], bandpass[1], fir_design='firwin', verbose=False)
    
    return raw


def reject_artifacts(
    raw: mne.io.Raw,
    threshold: float = None,
    epoch_length: float = 2.0
) -> Tuple[mne.io.Raw, float]:
    """
    Reject epochs exceeding amplitude threshold.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    threshold : float
        Amplitude threshold in volts
    epoch_length : float
        Epoch length in seconds
        
    Returns
    -------
    raw : mne.io.Raw
        Data with bad segments marked
    reject_ratio : float
        Fraction of data rejected
    """
    if threshold is None:
        threshold = PREPROC['artifact_threshold']
    
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_length * sfreq)
    n_epochs = data.shape[1] // samples_per_epoch
    
    n_rejected = 0
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch_data = data[:, start:end]
        if np.max(np.abs(epoch_data)) > threshold:
            n_rejected += 1
    
    reject_ratio = n_rejected / n_epochs if n_epochs > 0 else 0
    return raw, reject_ratio


def load_and_preprocess(
    subject: int,
    condition: str = 'rest',
    verbose: bool = False
) -> Optional[mne.io.Raw]:
    """
    Load and preprocess a single subject's EEG data.
    
    Parameters
    ----------
    subject : int
        Subject number (1-109)
    condition : str
        'rest' or 'task'
    verbose : bool
        Print progress messages
        
    Returns
    -------
    raw : mne.io.Raw or None
        Preprocessed data, or None if rejected
    """
    runs = PHYSIONET['rest_runs'] if condition == 'rest' else PHYSIONET['task_runs']
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = load_physionet_subject(subject, runs)
            raw = preprocess_raw(raw)
            raw, reject_ratio = reject_artifacts(raw)
            
            if reject_ratio > PREPROC['reject_ratio']:
                if verbose:
                    print(f"  Subject {subject}: rejected ({reject_ratio:.1%} artifacts)")
                return None
            
            if verbose:
                print(f"  Subject {subject}: OK ({reject_ratio:.1%} artifacts)")
            return raw
            
    except Exception as e:
        if verbose:
            print(f"  Subject {subject}: error ({e})")
        return None


def get_valid_subjects(
    condition: str = 'rest',
    max_subjects: int = None,
    verbose: bool = True
) -> List[int]:
    """
    Get list of subjects passing quality control.
    
    Parameters
    ----------
    condition : str
        'rest' or 'task'
    max_subjects : int
        Maximum number of subjects to check
    verbose : bool
        Print progress
        
    Returns
    -------
    valid : list of int
        Subject numbers passing QC
    """
    n_subjects = max_subjects or PHYSIONET['n_subjects']
    valid = []
    
    if verbose:
        print(f"Checking subjects for {condition} condition...")
    
    for subj in range(1, n_subjects + 1):
        raw = load_and_preprocess(subj, condition, verbose=False)
        if raw is not None:
            valid.append(subj)
    
    if verbose:
        print(f"Valid subjects: {len(valid)}/{n_subjects}")
    
    return valid
