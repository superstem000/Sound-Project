import sounddevice as sd
import numpy as np
import threading
import scipy.io.wavfile as wav
import time
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal import butter, filtfilt, correlate

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

def bandpass_filter(signal, fs, lowcut, highcut):
    # Create a bandpass filter with given low and high cutoff frequencies
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')  # 4th order filter
    
    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def compute_detailed_tdoa_large_window(file1, file2, fs, window_size, energy_threshold=0.3):
    """
    Compute detailed TDOA by cross-correlating larger overlapping windows with a shift of 1 sample.
    Compare each window in the first signal (100 consecutive time frames) to all windows in the second signal.
    """
    # Load audio files
    fs1, sig1 = wav.read(file1)
    fs2, sig2 = wav.read(file2)
    
    # Ensure sampling rates match
    assert fs1 == fs2 == fs, "Sampling rates do not match!"

    # Convert to mono if stereo
    if sig1.ndim > 1: sig1 = sig1[:, 0]
    if sig2.ndim > 1: sig2 = sig2[:, 0]
    
    # Normalize signals to avoid amplitude bias
    sig1 = sig1.astype(np.float32)
    sig2 = sig2.astype(np.float32)
    sig1 /= np.max(np.abs(sig1))
    sig2 /= np.max(np.abs(sig2))

    # Apply bandpass filter to both signals
    sig1 = bandpass_filter(sig1, fs, 200, 3000)
    sig2 = bandpass_filter(sig2, fs, 200, 3000)
    
    detailed_tdoa = []
    
    # Compute STFT for both signals (will get complex frequency-domain representation)
    f1, t1, Zxx1 = stft(sig1, fs, nperseg=window_size, noverlap = window_size - 1)
    f2, t2, Zxx2 = stft(sig2, fs, nperseg=window_size, noverlap = window_size - 1)

    for i in range(0, len(t1) - window_size, window_size // 10):  # Loop from time 0 to end minus window_size
        # Create window of 100 consecutive time frames
        window1 = Zxx1[:, i:i + window_size]  # (frequency bins, 100 time frames)
        energy1 = np.sum(np.abs(window1) ** 2)

        best_shift = None
        max_corr = -np.inf  # Start with a very low correlation

        # Loop through possible windows in the second signal, constrained to those within 48 samples of the current window in the first signal
        start_j = max(0, i - 60)  # Start 48 samples before i
        end_j = min(len(t2) - window_size, i + 60)  # End 48 samples after i
        
        for j in range(start_j, end_j):  # Loop through time frames in second signal within 48 samples of i
            window2 = Zxx2[:, j:j + window_size]  # (frequency bins, 100 time frames)
            window2_compare = Zxx2[:, j - window_size // 4: j + window_size // 4 + window_size]

            #window1 /= np.max(np.abs(window1))  # Normalize each window
            #window2 /= np.max(np.abs(window2))  # Normalize each window

            # Compute the energy of the signal in the current window
            energy2 = np.sum(np.abs(window2) ** 2)

            # Check if both windows have enough energy
            if energy1 > energy_threshold and energy2 > energy_threshold:
                # Compare the two windows directly (without shifting)
                cross_corr = correlate(window1, window2, mode='valid')

                # Average the cross-correlation values (you can also explore other ways to summarize it)
                similarity = np.mean(cross_corr)

                # If this similarity is the highest we've found, save the shift
                if similarity > max_corr:
                    max_corr = similarity
                    best_shift = j - i  # The difference between the current positions of i and j

        # Store the best shift (i - j) for the current window of the first signal
        if best_shift is not None:
            detailed_tdoa.append(best_shift / 48000)  # This is the relative shift (i - j)
        else:
            detailed_tdoa.append(0)
    
    return np.array(detailed_tdoa)

def find_best_tdoa(detailed_tdoa, method='mean'):
    """
    Calculate the 'best' TDOA based on a chosen method.
    
    Args:
    - detailed_tdoa (numpy array): Array of detailed TDOA values.
    - method (str): Method to calculate the best TDOA. Options are 'mean', 'median', or 'max'. Default is 'mean'.
    
    Returns:
    - best_tdoa (float): The best TDOA value based on the chosen method.
    """
    
    if method == 'mean':
        best_tdoa = np.mean(detailed_tdoa)
    elif method == 'median':
        best_tdoa = np.median(detailed_tdoa)
    elif method == 'max':
        best_tdoa = np.max(detailed_tdoa)
    elif method == 'min':
        best_tdoa = np.min(detailed_tdoa)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'mean', 'median', 'max', or 'min'.")
    
    return best_tdoa

def visualize_detailed_tdoa_columns(detailed_tdoa):
    """ Plots the given data over time, assuming each index represents 1/fs seconds. """
    time_axis = np.arange(len(detailed_tdoa)) / fs  # Convert index to time in seconds

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, detailed_tdoa, marker=".", linestyle="-", markersize=2, label="TDOA Values")
    plt.xlabel("Time (seconds)")
    plt.ylabel("TDOA Value")
    plt.title("TDOA Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    #threads = [threading.Thread(target=record_mic, args=(i,)) for i in device_indices]
    #for t in threads: t.start()
    #for t in threads: t.join()
    #print("All recordings completed.")

    detail = compute_detailed_tdoa_large_window("mic_1.wav", "mic_2.wav", fs=48000, window_size=500)
    print(detail)
    visualize_detailed_tdoa_columns(detail)

    tdoa = find_best_tdoa(detail)
    print(tdoa)

if __name__ == '__main__':
    main()