import sounddevice as sd
import numpy as np
import threading
import scipy.io.wavfile as wav
from scipy.signal import correlate, stft, istft
from scipy.signal import butter, filtfilt
from numpy import hamming
import time

fs = 48000  # Sampling rate
duration = 5  # Recording duration in seconds
device_indices = [1, 2, 3, 4]  # Replace with actual device indices
start_delay = 1  # Delay to synchronize start time

# Synchronization: Align all recordings to start at the same time
start_time = time.time() + start_delay

def record_mic(index):
    while time.time() < start_time:
        pass  # Busy wait until exact start time
    
    print("Start recording...")
    
    # Record audio
    data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32', device=index)
    sd.wait()

    # Clip the data between -1 and 1 (to avoid distortion)
    data = np.clip(data, -1, 1)

    # Save the file with proper scaling for 16-bit WAV
    wav.write(f"mic_{index}.wav", fs, (data * 32767).astype(np.int16))  # 16-bit WAV file
    print(f"Mic {index} saved.")

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

def compute_tdoa(file1, file2, mic_distance, fs=48000, threshold=0.05, max_tdoa_ms=1):
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

    # Compute STFT for both signals (will get complex frequency-domain representation)
    f1, t1, Zxx1 = stft(sig1, fs, nperseg=fs, noverlap = 0)
    f2, t2, Zxx2 = stft(sig2, fs, nperseg=fs, noverlap = 0)

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
    best_lag = np.argmax(limited_corr) + search_range[0] - center_index

    # Convert delay to time
    tdoa = best_lag / fs
    return tdoa

def main():
    threads = [threading.Thread(target=record_mic, args=(i,)) for i in device_indices]
    for t in threads: t.start()
    for t in threads: t.join()
    print("All recordings completed.")

    tdoa_12 = compute_tdoa("mic_1.wav", "mic_2.wav", 0.25)
    tdoa_34 = compute_tdoa("mic_3.wav", "mic_4.wav", 0.25)
    print(f"TDOA for Mic Pair (1-2): {tdoa_12}")
    print(f"TDOA for Mic Pair (3-4): {tdoa_34}")



if __name__ == '__main__':
    main()
