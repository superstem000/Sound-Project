import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import correlate, stft, istft
from scipy.signal import butter, filtfilt
from numpy import hamming

def filter_spectrogram(sig, fs, threshold=0.01):
    """ Applies a spectrogram filter to remove low-energy components. """
    f, t, Zxx = stft(sig, fs, nperseg=1024)
    magnitude = np.abs(Zxx)
    
    # Zero out low-energy frequency components
    Zxx[magnitude < threshold * np.max(magnitude)] = 0
    
    # Reconstruct signal
    _, filtered_signal = istft(Zxx, fs)
    return filtered_signal

def bandpass_filter(signal, fs, lowcut, highcut):
    # Create a bandpass filter with given low and high cutoff frequencies
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')  # 4th order filter
    
    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def plot_correlation(file1, file2, fs=48000, max_tdoa_ms=5, threshold=0.05):
    # Load audio files
    fs1, sig1 = wav.read(file1)
    fs2, sig2 = wav.read(file2)
    
    # Ensure sampling rates match
    assert fs1 == fs2 == fs, "Sampling rates do not match!"

    # Convert to mono if stereo
    if sig1.ndim > 1: sig1 = sig1[:, 0]
    if sig2.ndim > 1: sig2 = sig2[:, 0]

    # Normalize signals (avoids amplitude bias)
    sig1 = sig1.astype(np.float32)
    sig2 = sig2.astype(np.float32)
    sig1 /= np.max(np.abs(sig1))
    sig2 /= np.max(np.abs(sig2))

    # Apply bandpass filter to both signals
    sig1 = bandpass_filter(sig1, fs, 300, 3000)
    sig2 = bandpass_filter(sig2, fs, 300, 3000)
    
    # Apply spectrogram filtering
    
    sig1 = filter_spectrogram(sig1, fs, threshold)
    sig2 = filter_spectrogram(sig2, fs, threshold)
    
    # Compute cross-correlation
    corr = correlate(sig1, sig2, mode="full")
    window = np.hamming(len(corr))  # Apply Hamming window
    corr = corr * window

    #threshold_value = np.max(corr) * 0.5  # Example: only consider peaks above 50% of the max
    #corr[corr < threshold_value] = 0

    # Limit search window to ±max_tdoa_ms
    max_lag = int((max_tdoa_ms / 1000) * fs)  # Convert ms to samples
    center_index = len(sig1) - 1
    search_range = (center_index - max_lag, center_index + max_lag)

    # Find peak only in the limited window
    limited_corr = corr[search_range[0]:search_range[1]]

    # Create time axis for plotting (in seconds)
    # Shift the time axis to reflect both negative and positive lags
    time = np.arange(-len(limited_corr) / 2, len(limited_corr) / 2) / fs

    # Plot the cross-correlation
    plt.figure(figsize=(10, 6))
    plt.plot(time, limited_corr)
    plt.title("Cross-Correlation between Audio Files")
    plt.xlabel("Time (s)")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.show()

# Example usage
plot_correlation('mic_3.wav', 'mic_4.wav')
