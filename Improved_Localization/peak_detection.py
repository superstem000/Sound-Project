import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

def find_peak(signal, fs, peak_window=1.5e-3):
    # Find the peak (amplitude maximum) in the signal
    peaks, _ = find_peaks(signal)
    
    if len(peaks) == 0:
        return None, None
    
    # Choose the peak with the maximum amplitude
    peak_idx = peaks[np.argmax(signal[peaks])]
    
    # Convert the peak index to time (in seconds)
    peak_time = peak_idx / fs
    
    # Define the search window in terms of samples based on peak_window (±1.5ms)
    search_window_samples = int(peak_window * fs)
    
    return peak_idx, peak_time, search_window_samples

def get_tdoa(file1, file2):
    # Read the .wav files
    fs1, sig1 = wavfile.read(file1)
    fs2, sig2 = wavfile.read(file2)

    # Ensure both signals are mono (1 channel)
    if sig1.ndim > 1:
        sig1 = sig1[:, 0]
    if sig2.ndim > 1:
        sig2 = sig2[:, 0]

    # Find peak in the first signal
    peak_idx1, peak_time1, search_window_samples = find_peak(sig1, fs1)
    if peak_idx1 is None:
        raise ValueError("No peaks found in the first signal.")

    print(f"Peak in the first signal at {peak_time1:.6f} seconds (index {peak_idx1})")

    # Search for the corresponding peak in the second signal within the ±1.5ms window
    start_idx = max(0, peak_idx1 - search_window_samples)
    end_idx = min(len(sig2), peak_idx1 + search_window_samples)

    # Extract the search window from the second signal
    sig2_window = sig2[start_idx:end_idx]

    # Find peak in the second signal within the window
    peak_idx2 = np.argmax(sig2_window) + start_idx
    peak_time2 = peak_idx2 / fs2

    print(f"Peak in the second signal at {peak_time2:.6f} seconds (index {peak_idx2})")

    # Calculate TDOA (in seconds)
    tdoa = peak_time2 - peak_time1
    print(f"TDOA between the two signals: {tdoa:.6f} seconds")

    return tdoa

# Example usage
file1 = "mic_3.wav"
file2 = "mic_4.wav"

tdoa = get_tdoa(file1, file2)
