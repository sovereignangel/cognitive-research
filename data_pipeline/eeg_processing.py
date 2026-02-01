"""
EEG Signal Processing Utilities
===============================
Computes features for consciousness research analysis.
Includes frequency band analysis, asymmetry metrics, and state indicators.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import entropy


# Frequency band definitions (Hz)
FREQUENCY_BANDS = {
    'delta': (0.5, 4),    # Deep sleep, unconscious
    'theta': (4, 8),      # Drowsy, meditation, memory
    'alpha': (8, 13),     # Relaxed, calm awareness
    'beta': (13, 30),     # Active thinking, focus
    'gamma': (30, 100)    # Higher cognition, perception
}

# Channel indices for Muse (TP9, AF7, AF8, TP10)
FRONTAL_LEFT = 1   # AF7
FRONTAL_RIGHT = 2  # AF8
TEMPORAL_LEFT = 0  # TP9
TEMPORAL_RIGHT = 3 # TP10


def compute_band_powers(
    eeg_data: np.ndarray,
    sample_rate: int = 256,
    window_seconds: float = 2.0
) -> Dict[str, np.ndarray]:
    """
    Compute power in each frequency band for each channel.

    Args:
        eeg_data: Array of shape (n_samples, n_channels)
        sample_rate: Sampling rate in Hz
        window_seconds: Window size for spectral analysis

    Returns:
        Dict mapping band name to array of shape (n_channels,)
    """
    n_channels = eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1
    if n_channels == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    band_powers = {band: np.zeros(n_channels) for band in FREQUENCY_BANDS}

    nperseg = min(int(window_seconds * sample_rate), len(eeg_data))

    for ch in range(n_channels):
        freqs, psd = signal.welch(
            eeg_data[:, ch],
            fs=sample_rate,
            nperseg=nperseg
        )

        for band_name, (low, high) in FREQUENCY_BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_powers[band_name][ch] = np.mean(psd[idx]) if np.any(idx) else 0

    return band_powers


def compute_relative_band_powers(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> Dict[str, np.ndarray]:
    """
    Compute relative band powers (normalized to total power).

    Returns:
        Dict mapping band name to array of relative powers (0-1)
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)

    # Sum total power across bands
    total_power = sum(band_powers.values())
    total_power = np.where(total_power > 0, total_power, 1)  # Avoid division by zero

    return {
        band: power / total_power
        for band, power in band_powers.items()
    }


def compute_alpha_asymmetry(
    eeg_data: np.ndarray,
    sample_rate: int = 256,
    frontal: bool = True
) -> float:
    """
    Compute frontal or temporal alpha asymmetry.

    Positive values = more left activation (approach motivation)
    Negative values = more right activation (withdrawal motivation)

    Args:
        eeg_data: Array of shape (n_samples, 4) for Muse channels
        sample_rate: Sampling rate in Hz
        frontal: If True, use frontal channels (AF7/AF8), else temporal (TP9/TP10)

    Returns:
        Alpha asymmetry score: ln(right_alpha) - ln(left_alpha)
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)
    alpha = band_powers['alpha']

    if frontal:
        left_idx, right_idx = FRONTAL_LEFT, FRONTAL_RIGHT
    else:
        left_idx, right_idx = TEMPORAL_LEFT, TEMPORAL_RIGHT

    left_power = alpha[left_idx]
    right_power = alpha[right_idx]

    # Asymmetry formula: ln(right) - ln(left)
    # More positive = more left activation (right alpha suppression)
    asymmetry = np.log(right_power + 1e-10) - np.log(left_power + 1e-10)

    return float(asymmetry)


def compute_engagement_index(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> float:
    """
    Compute engagement/alertness index: beta / (alpha + theta).

    Higher values indicate more cognitive engagement and alertness.
    Lower values indicate relaxation or drowsiness.
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)

    beta_power = np.mean(band_powers['beta'])
    alpha_power = np.mean(band_powers['alpha'])
    theta_power = np.mean(band_powers['theta'])

    denominator = alpha_power + theta_power
    if denominator < 1e-10:
        return 0.0

    return float(beta_power / denominator)


def compute_relaxation_index(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> float:
    """
    Compute relaxation index: alpha / beta.

    Higher values indicate more relaxation.
    Lower values indicate more active/stressed state.
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)

    alpha_power = np.mean(band_powers['alpha'])
    beta_power = np.mean(band_powers['beta'])

    if beta_power < 1e-10:
        return 0.0

    return float(alpha_power / beta_power)


def compute_meditation_depth(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> float:
    """
    Compute meditation depth indicator: (alpha + theta) / beta.

    Higher values suggest deeper meditative states.
    Based on research showing increased alpha/theta during meditation.
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)

    alpha_power = np.mean(band_powers['alpha'])
    theta_power = np.mean(band_powers['theta'])
    beta_power = np.mean(band_powers['beta'])

    if beta_power < 1e-10:
        return 0.0

    return float((alpha_power + theta_power) / beta_power)


def compute_focus_index(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> float:
    """
    Compute focus/concentration index: (beta + gamma) / (delta + theta).

    Higher values indicate focused concentration.
    Based on high-frequency activity during cognitive tasks.
    """
    band_powers = compute_band_powers(eeg_data, sample_rate)

    high_freq = np.mean(band_powers['beta']) + np.mean(band_powers['gamma'])
    low_freq = np.mean(band_powers['delta']) + np.mean(band_powers['theta'])

    if low_freq < 1e-10:
        return 0.0

    return float(high_freq / low_freq)


def compute_spectral_entropy(
    eeg_data: np.ndarray,
    sample_rate: int = 256
) -> np.ndarray:
    """
    Compute spectral entropy for each channel.

    Lower entropy = more organized/coherent brain activity
    Higher entropy = more random/chaotic activity
    """
    n_channels = eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1
    if n_channels == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    entropies = np.zeros(n_channels)

    for ch in range(n_channels):
        freqs, psd = signal.welch(eeg_data[:, ch], fs=sample_rate)

        # Normalize to probability distribution
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            entropies[ch] = entropy(psd_norm)

    return entropies


def detect_artifacts(
    eeg_data: np.ndarray,
    amplitude_threshold: float = 200.0,
    flat_threshold: float = 1.0
) -> np.ndarray:
    """
    Detect artifact samples (blinks, muscle activity, bad contact).

    Args:
        eeg_data: Array of shape (n_samples, n_channels)
        amplitude_threshold: Max amplitude in microvolts
        flat_threshold: Min std for flat signal detection

    Returns:
        Boolean array of shape (n_samples,) where True = artifact
    """
    if len(eeg_data.shape) == 1:
        eeg_data = eeg_data.reshape(-1, 1)

    # High amplitude artifacts (blinks, muscle)
    max_amplitude = np.max(np.abs(eeg_data), axis=1)
    high_amp = max_amplitude > amplitude_threshold

    # Flat signal detection (bad electrode contact)
    # Use rolling window std
    window = 50
    flat_signal = np.zeros(len(eeg_data), dtype=bool)

    for i in range(0, len(eeg_data) - window, window):
        chunk = eeg_data[i:i+window]
        if np.all(np.std(chunk, axis=0) < flat_threshold):
            flat_signal[i:i+window] = True

    return high_amp | flat_signal


def compute_session_features(
    eeg_data: np.ndarray,
    timestamps: np.ndarray,
    sample_rate: int = 256,
    window_seconds: float = 30.0,
    overlap: float = 0.5
) -> List[Dict]:
    """
    Compute features for an entire session in sliding windows.

    Args:
        eeg_data: Array of shape (n_samples, n_channels)
        timestamps: Array of timestamps
        sample_rate: Sampling rate in Hz
        window_seconds: Window size for feature computation
        overlap: Overlap fraction between windows (0-1)

    Returns:
        List of feature dictionaries, one per window
    """
    n_samples = len(eeg_data)
    window_samples = int(window_seconds * sample_rate)
    step_samples = int(window_samples * (1 - overlap))

    features = []

    for start in range(0, n_samples - window_samples, step_samples):
        end = start + window_samples
        window_data = eeg_data[start:end]
        window_time = timestamps[start] if start < len(timestamps) else 0

        # Skip windows with too many artifacts
        artifacts = detect_artifacts(window_data)
        artifact_ratio = np.mean(artifacts)
        if artifact_ratio > 0.3:
            continue

        # Compute all features
        band_powers = compute_band_powers(window_data, sample_rate)
        relative_powers = compute_relative_band_powers(window_data, sample_rate)

        feature_dict = {
            'timestamp': float(window_time),
            'window_start_sample': start,
            'window_end_sample': end,
            'artifact_ratio': float(artifact_ratio),

            # Absolute band powers (averaged across channels)
            'delta_power': float(np.mean(band_powers['delta'])),
            'theta_power': float(np.mean(band_powers['theta'])),
            'alpha_power': float(np.mean(band_powers['alpha'])),
            'beta_power': float(np.mean(band_powers['beta'])),
            'gamma_power': float(np.mean(band_powers['gamma'])),

            # Relative band powers
            'delta_relative': float(np.mean(relative_powers['delta'])),
            'theta_relative': float(np.mean(relative_powers['theta'])),
            'alpha_relative': float(np.mean(relative_powers['alpha'])),
            'beta_relative': float(np.mean(relative_powers['beta'])),
            'gamma_relative': float(np.mean(relative_powers['gamma'])),

            # Derived state metrics
            'alpha_asymmetry_frontal': compute_alpha_asymmetry(window_data, sample_rate, frontal=True),
            'alpha_asymmetry_temporal': compute_alpha_asymmetry(window_data, sample_rate, frontal=False),
            'engagement_index': compute_engagement_index(window_data, sample_rate),
            'relaxation_index': compute_relaxation_index(window_data, sample_rate),
            'meditation_depth': compute_meditation_depth(window_data, sample_rate),
            'focus_index': compute_focus_index(window_data, sample_rate),

            # Complexity
            'spectral_entropy': float(np.mean(compute_spectral_entropy(window_data, sample_rate)))
        }

        features.append(feature_dict)

    return features


def analyze_session(file_path: str) -> Dict:
    """
    Analyze a complete EEG session from file.

    Args:
        file_path: Path to .npz session file

    Returns:
        Dict with session analysis including:
        - summary statistics
        - time series of features
        - quality metrics
    """
    # Load session data
    data = np.load(file_path, allow_pickle=True)

    eeg = data['eeg']
    timestamps = data['eeg_timestamps']
    sample_rate = int(data['eeg_sample_rate'])

    # Compute windowed features
    features = compute_session_features(eeg, timestamps, sample_rate)

    if not features:
        return {'error': 'No valid windows (too many artifacts)'}

    # Compute summary statistics
    summary = {}
    for key in features[0].keys():
        if key not in ['timestamp', 'window_start_sample', 'window_end_sample']:
            values = [f[key] for f in features]
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))

    # Overall quality
    artifact_ratios = [f['artifact_ratio'] for f in features]
    summary['overall_quality'] = float(100 * (1 - np.mean(artifact_ratios)))
    summary['n_windows'] = len(features)
    summary['duration_seconds'] = len(eeg) / sample_rate

    # Linked health data
    linked_sleep = int(data.get('linked_sleep_score', -1))
    summary['linked_sleep_score'] = linked_sleep if linked_sleep >= 0 else None

    return {
        'summary': summary,
        'features_timeseries': features,
        'session_id': str(data['session_id']),
        'start_time': str(data['start_time'])
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="EEG Processing CLI")
    parser.add_argument("command", choices=["analyze", "demo"])
    parser.add_argument("--file", type=str, help="Session file path")
    parser.add_argument("--output", type=str, help="Output JSON path")

    args = parser.parse_args()

    if args.command == "analyze":
        if not args.file:
            print("Error: --file required for analyze command")
            exit(1)

        print(f"Analyzing session: {args.file}")
        result = analyze_session(args.file)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print("\nSession Summary:")
            print("-" * 40)
            for key, value in result['summary'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    elif args.command == "demo":
        # Generate synthetic data for testing
        print("Generating synthetic EEG data...")

        sample_rate = 256
        duration = 60  # 1 minute
        n_samples = sample_rate * duration
        n_channels = 4

        # Create synthetic signal with alpha oscillation
        t = np.linspace(0, duration, n_samples)
        alpha_freq = 10  # Hz

        eeg = np.zeros((n_samples, n_channels))
        for ch in range(n_channels):
            # Base alpha oscillation
            eeg[:, ch] = 20 * np.sin(2 * np.pi * alpha_freq * t)
            # Add some noise
            eeg[:, ch] += np.random.randn(n_samples) * 5
            # Add beta activity
            eeg[:, ch] += 5 * np.sin(2 * np.pi * 20 * t)

        timestamps = t

        print(f"\nComputing features for {duration}s of synthetic data...")
        features = compute_session_features(eeg, timestamps, sample_rate, window_seconds=10)

        print(f"\nComputed {len(features)} feature windows:")
        print("-" * 50)

        # Show first window
        if features:
            print("\nFirst window features:")
            for key, value in features[0].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
