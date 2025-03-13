import numpy as np
import wave
import os
import json
import glob
from scipy.io import wavfile
from itertools import combinations

def compute_tdoa_with_interpolation(sig1, sig2, fs=16000, max_tdoa_ms=5):
    """
    Compute TDOA between two signals using GCC-PHAT with parabolic interpolation.
    """
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
    valid_lags = np.arange(start_idx - mid_point, end_idx - mid_point)

    # Find the peak index within the valid correlation range
    peak_idx = np.argmax(valid_correlation)
    
    # Get the peak lag in samples
    coarse_lag = valid_lags[peak_idx]
    
    # Implement parabolic interpolation for sub-sample precision
    if 0 < peak_idx < len(valid_correlation) - 1:
        # Get values around the peak
        y_1 = valid_correlation[peak_idx - 1]
        y0 = valid_correlation[peak_idx]
        y1 = valid_correlation[peak_idx + 1]
        
        # Parabolic interpolation to find the refined peak position
        offset = 0.5 * (y_1 - y1) / (y_1 - 2*y0 + y1)
        
        # If the parabola is inverted, discard the interpolation
        if y_1 - 2*y0 + y1 < 0:
            # Fine peak position
            fine_lag = coarse_lag + offset
        else:
            fine_lag = coarse_lag
    else:
        fine_lag = coarse_lag
    
    # TDOA calculation (in seconds)
    tdoa = fine_lag / fs  # TDOA in seconds
    
    # Calculate correlation peak height as confidence measure
    confidence = valid_correlation[peak_idx]
    
    return tdoa, confidence

class MicArray:
    def __init__(self):
        # Assuming ReSpeaker 4-mic array, positioned at 45°, 135°, 225°, 315°
        # Distance from center to each mic is 45.7mm (or 4.57cm)
        self.radius = 0.0457  # meters
        
        # Mic positions in radians (45°, 135°, 225°, 315° converted to radians)
        self.mic_angles = np.array([45, 135, 225, 315]) * np.pi / 180
        
        # Calculate x, y coordinates for each mic
        self.mic_positions = np.zeros((4, 2))
        for i in range(4):
            self.mic_positions[i, 0] = self.radius * np.cos(self.mic_angles[i])  # x
            self.mic_positions[i, 1] = self.radius * np.sin(self.mic_angles[i])  # y
        
        # Speed of sound in air (m/s)
        self.sound_speed = 343.0

def calc_expected_tdoa(angle_rad, mic_positions, sound_speed):
    """
    Calculate the expected TDOA for a sound source at a given angle.
    """
    # Unit vector pointing to sound source direction
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Calculate the projection of mic positions onto the direction vector
    projections = np.dot(mic_positions, direction)
    
    # Calculate TDOAs for all mic pairs
    mic_pairs = list(combinations(range(len(mic_positions)), 2))
    expected_tdoas = {}
    
    for i, j in mic_pairs:
        # Time difference = difference in projections / speed of sound
        tdoa = (projections[i] - projections[j]) / sound_speed
        expected_tdoas[(i, j)] = tdoa
        
    return expected_tdoas

def calculate_doa_improved(mic_signals, fs, mic_array):
    """
    Calculate the direction of arrival (DOA) using improved TDOA from multiple mic pairs.
    """
    # Calculate TDOAs for all mic pairs using improved GCC-PHAT
    mic_pairs = list(combinations(range(len(mic_signals)), 2))
    measured_tdoas = {}
    confidences = {}
    
    for i, j in mic_pairs:
        tdoa, conf = compute_tdoa_with_interpolation(mic_signals[i], mic_signals[j], fs=fs)
        measured_tdoas[(i, j)] = tdoa
        confidences[(i, j)] = conf
        print(f"TDOA between mic {i+1} and mic {j+1}: {tdoa*1000:.3f} ms (confidence: {conf:.2f})")
    
    # Search for the angle that best matches the measured TDOAs
    angles = np.linspace(0, 2*np.pi, 720)  # Search in 0.5° increments for higher precision
    min_error = float('inf')
    best_angle = 0
    
    for angle in angles:
        # Calculate expected TDOAs for this angle
        expected = calc_expected_tdoa(angle, mic_array.mic_positions, mic_array.sound_speed)
        
        # Calculate weighted error between measured and expected TDOAs
        error = 0
        for pair in measured_tdoas:
            # Weight the error by the confidence
            pair_error = (measured_tdoas[pair] - expected[pair])**2
            weighted_error = pair_error * confidences[pair]
            error += weighted_error
        
        # Update best angle if this is better
        if error < min_error:
            min_error = error
            best_angle = angle
    
    # Calculate the overall confidence in the DOA estimate
    # Lower min_error means higher confidence
    if min_error < 1e-10:
        confidence = 1.0
    else:
        confidence = 1.0 / (1.0 + min_error * 10000)  # Scale to 0-1 range
    
    # Convert to degrees (0-360°)
    doa_degrees = (best_angle * 180 / np.pi) % 360
    
    return doa_degrees, confidence

def find_wav_files(dir_path, device_index, chunk_number):
    """
    Find WAV files for a specific device and chunk.
    """
    mic_files = []
    
    # First try using the timestamp file
    timestamp_file = f"{dir_path}/device{device_index}_chunk{chunk_number}_timestamps.json"
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            timestamp_data = json.load(f)
        
        mic_data = {}
        for file_info in timestamp_data.get("files", []):
            if file_info.get("channel_type", "").startswith("mic"):
                mic_number = int(file_info["channel_type"][3:])
                if 1 <= mic_number <= 4:
                    mic_data[mic_number] = file_info["filename"]
        
        for i in range(1, 5):
            if i in mic_data:
                mic_files.append(os.path.join(dir_path, mic_data[i]))
    
    # If that doesn't work, try pattern matching
    if len(mic_files) != 4:
        pattern = f"device{device_index}_mic*_chunk{chunk_number}.wav"
        matching_files = sorted(glob.glob(os.path.join(dir_path, pattern)))
        if len(matching_files) >= 4:
            mic_files = matching_files[:4]
    
    if len(mic_files) != 4:
        raise ValueError(f"Could not find all 4 microphone files for device {device_index}, chunk {chunk_number}")
    
    return mic_files

def process_chunk(dir_path, device_index, chunk_number):
    """
    Process a single chunk to calculate DOA.
    """
    print(f"Processing device {device_index}, chunk {chunk_number}")
    
    # Find and load WAV files
    mic_files = find_wav_files(dir_path, device_index, chunk_number)
    print(f"Found microphone files: {[os.path.basename(f) for f in mic_files]}")
    
    mic_signals = []
    fs = None
    
    for file_path in mic_files:
        sample_rate, signal = wavfile.read(file_path)
        if fs is None:
            fs = sample_rate
        mic_signals.append(signal)
    
    # Create mic array
    mic_array = MicArray()
    
    # Calculate DOA
    doa, confidence = calculate_doa_improved(mic_signals, fs, mic_array)
    print(f"DOA: {doa:.2f}° (confidence: {confidence:.2f})")
    
    return doa, confidence

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python doa.py <directory> <device_index> <chunk_number>")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    device_index = int(sys.argv[2])
    chunk_number = int(sys.argv[3])
    
    try:
        doa, confidence = process_chunk(dir_path, device_index, chunk_number)
        print(f"\nFinal result: DOA = {doa:.2f}°, Confidence = {confidence:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()