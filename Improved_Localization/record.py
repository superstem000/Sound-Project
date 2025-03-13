import sounddevice as sd
import numpy as np
import threading
import scipy.io.wavfile as wav
from scipy.signal import correlate, stft, istft, correlation_lags
from scipy.signal import butter, filtfilt
from numpy import hamming
import time
import queue
import matplotlib.pyplot as plt

fs = 48000  # Sampling rate
duration = 6  # Recording duration in seconds
device_indices = [1, 2, 3, 4]  # Replace with actual device indices
start_delay = 1  # Delay to synchronize start time

# Synchronization: Align all recordings to start at the same time
start_time = time.time() + start_delay

def record_mic():
    # Create a list to store recordings
    recordings = [None] * len(device_indices)
    
    # Callback function to store recorded data
    def callback(indata, frames, time, status, device_idx):
        if status:
            print(f'Error in device {device_idx}: {status}')
        # Accumulate the data for each device
        if recordings[device_idx] is None:
            recordings[device_idx] = indata.copy()
        else:
            recordings[device_idx] = np.vstack((recordings[device_idx], indata))

    # Create input streams for all devices but don't start them yet
    streams = []
    for idx, device in enumerate(device_indices):
        stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=fs,
            callback=lambda *args, idx=idx: callback(*args, idx)
        )
        streams.append(stream)
    
    # Start all streams simultaneously
    for stream in streams:
        stream.start()
    
    # Wait for duration
    sd.sleep(int(duration * 1000))
    
    # Stop all streams
    for stream in streams:
        stream.stop()
        stream.close()

    # Save recordings to .wav files
    for idx, recording in enumerate(recordings):
        if recording is not None:
            filename = f"mic_{device_indices[idx]}.wav"
            wav.write(filename, fs, (recording * 32767).astype(np.int16))
            print(f"Saved recording from device {device_indices[idx]} to {filename}")

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

def compute_tdoa(sig1, sig2, fs=48000, max_tdoa_ms=5):
    # Convert to mono if stereo
    if sig1.ndim > 1: sig1 = sig1[:, 0]
    if sig2.ndim > 1: sig2 = sig2[:, 0]

    # Normalize signals
    sig1 = sig1.astype(np.float32) / (np.max(np.abs(sig1)) + 1e-10)
    sig2 = sig2.astype(np.float32) / (np.max(np.abs(sig2)) + 1e-10)

    # Compute FFT of both signals
    n = max(sig1.shape[0], sig2.shape[0])
    SIG = np.fft.fft(sig1, n=n)
    REFSIG = np.fft.fft(sig2, n=n)
    
    # Compute cross-power spectrum with PHAT weighting
    R = SIG * np.conj(REFSIG)  # Cross-power spectrum
    R_phat = np.where(np.abs(R) > 1e-7, R / np.abs(R), 0)  # PHAT normalization

    # Inverse FFT to get time-domain cross-correlation
    cross_corr = np.fft.ifft(R_phat).real  # Time-domain correlation

    # Find the peak in the cross-correlation signal within a certain range (max_tdoa_ms)
    max_lag_samples = int((max_tdoa_ms / 1000) * fs)  # max_tdoa_ms is in ms, fs is in Hz
    mid_point = len(cross_corr) // 2

    # Get the range of lags to consider
    start_idx = max(mid_point - max_lag_samples, 0)
    end_idx = min(mid_point + max_lag_samples + 1, len(cross_corr))

    # Extract the valid correlation (within the desired lag range)
    valid_correlation = cross_corr[start_idx:end_idx]

    # Create the corresponding valid lags (in samples)
    valid_lags = np.arange(-max_lag_samples, max_lag_samples + 1)

    # Find the peak index within the valid correlation range
    peak_idx = np.argmax(valid_correlation)

    # TDOA calculation (in seconds)
    tdoa = valid_lags[peak_idx] / fs  # TDOA in seconds

    #plt.figure(figsize=(10, 4))
    #plt.plot(valid_lags, valid_correlation)
    #plt.axvline(x=valid_lags[peak_idx], color='r', linestyle='--')
    #plt.xlabel('Lag (samples)')
    #plt.ylabel('Correlation')
    #plt.grid(True)
    #plt.show()

    return tdoa

def split_into_frames(signal, frame_size, fs):
    """Splits a signal into non-overlapping frames, including the last partial frame."""
    frame_samples = int(frame_size * fs)  # Convert seconds to samples
    
    frames = [signal[i : i + frame_samples] for i in range(0, 240000, frame_samples)]
    return np.array(frames, dtype=object)  # Use dtype=object to allow varying lengths

def compute_simple_tdoa(sig1, sig2, fs=48000, max_tdoa_ms=5):
    # Convert to mono if stereo
    if sig1.ndim > 1: sig1 = sig1[:, 0]
    if sig2.ndim > 1: sig2 = sig2[:, 0]

    # Normalize signals
    sig1 = sig1.astype(np.float32) / (np.max(np.abs(sig1)) + 1e-10)
    sig2 = sig2.astype(np.float32) / (np.max(np.abs(sig2)) + 1e-10)
    
    # Detect peak in sig1
    peak_idx_sig1 = np.argmax(np.abs(sig1))  # Find the maximum absolute value index

    # Calculate max lag in samples based on max_tdoa_ms
    max_lag_samples = int((max_tdoa_ms / 1000) * fs)  # Convert max TDOA in ms to samples
    
    # Define the search range in sig2 around the detected peak in sig1
    start_idx_sig2 = max(peak_idx_sig1 - max_lag_samples, 0)
    end_idx_sig2 = min(peak_idx_sig1 + max_lag_samples, len(sig2))

    # Extract the search window from sig2
    search_window_sig2 = sig2[start_idx_sig2:end_idx_sig2]
    
    # Find the peak in the search window of sig2
    peak_idx_sig2 = np.argmax(np.abs(search_window_sig2))  # Maximum in the defined region
    
    # Calculate the TDOA in samples
    tdoa_samples = peak_idx_sig2 - (peak_idx_sig1 - start_idx_sig2)

    # Convert TDOA to seconds
    tdoa_seconds = tdoa_samples / fs

    return tdoa_seconds

    

def main():
    print("start recording")
    record_mic()
    print("finished")

    fs1, sig1 = wav.read("mic_1.wav")
    fs2, sig2 = wav.read("mic_2.wav")
    fs3, sig3 = wav.read("mic_3.wav")
    fs4, sig4 = wav.read("mic_4.wav")

    plt.plot(sig1, label="Signal 1")
    plt.plot(sig2, label="Signal 2")
    plt.legend()
    #plt.xlim(44000, 47000)  # Set the x-axis limit to range from 0 to 20000
    plt.show()

    assert fs1 == fs2 == fs3 == fs4, "Sampling rates do not match!"

    # Convert to mono if stereo
    if sig1.ndim > 1: sig1 = sig1[:, 0]
    if sig2.ndim > 1: sig2 = sig2[:, 0]
    if sig3.ndim > 1: sig3 = sig3[:, 0]
    if sig4.ndim > 1: sig4 = sig4[:, 0]

    simple_tdoa1 = compute_simple_tdoa(sig1, sig2)
    simple_tdoa2 = compute_simple_tdoa(sig3, sig4)
    print(simple_tdoa1)
    print(simple_tdoa2)
    print("simple above")

    # Split into 50ms frames
    frame_size = 0.20  # 200ms
    frames1 = split_into_frames(sig1, frame_size, fs1)
    frames2 = split_into_frames(sig2, frame_size, fs1)
    frames3 = split_into_frames(sig3, frame_size, fs1)
    frames4 = split_into_frames(sig4, frame_size, fs1)

    # Compute TDOA for each frame
    tdoa_values1 = []
    tdoa_values2 = []
    #print(f"Frames1 count: {len(frames1)}, Frames2 count: {len(frames2)}, Frames3 count: {len(frames3)}, Frames4 count: {len(frames4)}")
    #frame_samples = int(frame_size * fs1)
    #print(f"Frame size in samples: {frame_samples}")
    #print(f"Signal length in samples: {len(sig1)}")
    #print(f"Expected number of frames: {len(sig1) // frame_samples}")
    for frame1, frame2, frame3, frame4 in zip(frames1, frames2, frames3, frames4):
        tdoa1 = compute_tdoa(frame1, frame2, fs1)  # Modify compute_tdoa to accept raw signals
        tdoa2 = compute_tdoa(frame3, frame4, fs1)
        tdoa_values1.append(tdoa1)
        tdoa_values2.append(tdoa2)

    mean_tdoa1 = np.median(tdoa_values1)
    mean_tdoa2 = np.median(tdoa_values2)
    print(mean_tdoa1)
    print(mean_tdoa2)
    
    #print("TDOA 1-2 values over time:", tdoa_values1)
    #print("TDOA 3-4 values over time:", tdoa_values2)


    



if __name__ == '__main__':
    main()
