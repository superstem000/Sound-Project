import numpy as np
import os
import json
import wave
import glob
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
from itertools import combinations
import math

# Import existing DOA calculation code (assuming doa.py is in the same directory)
from doa import compute_tdoa_with_interpolation, MicArray, calc_expected_tdoa, calculate_doa_improved, find_wav_files, process_chunk

def triangulate_sound_source(doa1, doa2, array_distance, room_height=None, room_width=None):
    """
    Triangulate the sound source position based on two DOA measurements.
    
    Parameters:
    - doa1: DOA from first array in degrees (0-360)
    - doa2: DOA from second array in degrees (0-360)
    - array_distance: Distance between the two arrays in meters
    - room_height, room_width: Optional room dimensions for plotting
    
    Returns:
    - (x, y): Estimated source position in meters
    """
    # Convert DOAs to radians
    doa1_rad = doa1 * np.pi / 180
    doa2_rad = doa2 * np.pi / 180
    
    # Define array positions (first array at origin, second array at (array_distance, 0))
    array1_pos = np.array([0, 0])
    array2_pos = np.array([array_distance, 0])
    
    # Calculate direction vectors from each array
    v1 = np.array([np.cos(doa1_rad), np.sin(doa1_rad)])
    v2 = np.array([np.cos(doa2_rad), np.sin(doa2_rad)])
    
    # Line equations: array_pos + t*v where t is a scalar
    # To find intersection, solve for t1, t2 where:
    # array1_pos + t1*v1 = array2_pos + t2*v2
    
    # Set up linear system:
    # t1*v1 - t2*v2 = array2_pos - array1_pos
    A = np.column_stack((v1, -v2))
    b = array2_pos - array1_pos
    
    # Check if we can solve the system (lines aren't parallel)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    if abs(det) < 1e-6:
        # Lines are nearly parallel, use midpoint of closest approach
        print("Warning: DOA lines are nearly parallel, using approximation")
        
        # Calculate parameters for closest approach
        v1_dot_v2 = np.dot(v1, v2)
        denom = 1 - v1_dot_v2**2
        
        if abs(denom) < 1e-6:
            # Truly parallel, use midpoint between arrays
            return (array_distance / 2, 0)
            
        t1 = np.dot(array2_pos - array1_pos, v1 - v2 * v1_dot_v2) / denom
        t2 = np.dot(array2_pos - array1_pos, v2 - v1 * v1_dot_v2) / denom
        
        p1 = array1_pos + t1 * v1
        p2 = array2_pos + t2 * v2
        
        # Midpoint between closest approach points
        position = (p1 + p2) / 2
    else:
        # Solve the linear system for intersection
        solution = np.linalg.solve(A, b)
        t1 = solution[0]
        
        # Calculate intersection point
        position = array1_pos + t1 * v1
    
    # Optional: Plot the triangulation
    if room_width is not None and room_height is not None:
        plt.figure(figsize=(10, 8))
        
        # Plot room boundaries
        plt.plot([0, room_width, room_width, 0, 0], [0, 0, room_height, room_height, 0], 'k-', alpha=0.3)
        
        # Plot array positions
        plt.scatter([array1_pos[0], array2_pos[0]], [array1_pos[1], array2_pos[1]], 
                   color='blue', s=100, marker='^', label='Mic Arrays')
        
        # Plot DOA lines
        line_length = max(room_width, room_height) * 1.5
        plt.plot([array1_pos[0], array1_pos[0] + line_length * v1[0]],
                [array1_pos[1], array1_pos[1] + line_length * v1[1]],
                'r--', alpha=0.5, label='DOA 1')
        plt.plot([array2_pos[0], array2_pos[0] + line_length * v2[0]],
                [array2_pos[1], array2_pos[1] + line_length * v2[1]],
                'g--', alpha=0.5, label='DOA 2')
        
        # Plot estimated position
        plt.scatter([position[0]], [position[1]], color='red', s=150, marker='o', label='Source')
        
        plt.title(f'Sound Source Triangulation\nDOA1={doa1:.1f}°, DOA2={doa2:.1f}°')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-0.5, room_width + 0.5)
        plt.ylim(-0.5, room_height + 0.5)
        plt.show()
    
    return position[0], position[1]

def localize_from_two_arrays(dir_path, device1_index, device2_index, chunk_number, array_distance, plot=False):
    """
    Localize a sound source using two ReSpeaker arrays.
    
    Parameters:
    - dir_path: Directory containing WAV files
    - device1_index, device2_index: Indices of the two ReSpeaker devices
    - chunk_number: Chunk to process
    - array_distance: Distance between arrays in meters
    - plot: Whether to plot the triangulation result
    
    Returns:
    - (x, y): Estimated source position in meters
    - (doa1, doa2): DOA estimates from each array
    - (conf1, conf2): Confidence scores from each array
    """
    # Calculate DOA for each array
    doa1, conf1 = process_chunk(dir_path, device1_index, chunk_number)
    doa2, conf2 = process_chunk(dir_path, device2_index, chunk_number)
    
    print(f"Array 1 (Device {device1_index}): DOA = {doa1:.2f}° (confidence: {conf1:.2f})")
    print(f"Array 2 (Device {device2_index}): DOA = {doa2:.2f}° (confidence: {conf2:.2f})")
    
    # Triangulate source position
    x, y = triangulate_sound_source(doa1, doa2, array_distance, 
                                    room_height=4.0, room_width=6.0 if plot else None)
    
    print(f"Estimated source position: ({x:.2f}, {y:.2f}) meters")
    
    return (x, y), (doa1, doa2), (conf1, conf2)

def location_based_beamforming(mic_signals, fs, mic_positions, source_position):
    """
    Perform delay-and-sum beamforming using the exact source location for more accurate delays.
    
    Parameters:
    - mic_signals: List of audio signals from microphones
    - fs: Sampling frequency
    - mic_positions: Array of mic positions [(x1,y1), (x2,y2), ...]
    - source_position: (x, y) position of the sound source
    
    Returns:
    - Enhanced audio signal
    """
    # Convert source position to numpy array if it's not already
    source_position = np.array(source_position)
    
    # Calculate the Euclidean distance from each mic to the source
    distances = np.array([np.linalg.norm(mic_pos - source_position) for mic_pos in mic_positions])
    
    # Calculate relative delays (in samples)
    # Subtract the minimum distance so the first arrival has zero delay
    delays = (distances - np.min(distances)) / 343.0 * fs
    
    # Find the maximum delay
    max_delay = int(np.ceil(np.max(delays)))
    
    # Initialize output with padding for the delays
    if len(mic_signals[0].shape) > 1:
        # Stereo signals, use first channel
        output_length = max(sig.shape[0] for sig in mic_signals) + max_delay
        output = np.zeros(output_length)
        for i, signal in enumerate(mic_signals):
            # Apply delay
            delay_samples = int(np.round(delays[i]))
            output[delay_samples:delay_samples + signal.shape[0]] += signal[:, 0]
    else:
        # Mono signals
        output_length = max(len(sig) for sig in mic_signals) + max_delay
        output = np.zeros(output_length)
        for i, signal in enumerate(mic_signals):
            # Apply delay
            delay_samples = int(np.round(delays[i]))
            output[delay_samples:delay_samples + len(signal)] += signal
    
    # Normalize output
    output = output / len(mic_signals)
    
    return output

# Update the beamform_from_all_mics function to use the new beamforming approach
def beamform_from_all_mics(dir_path, device1_index, device2_index, chunk_number, source_position, array_distance):
    """
    Perform beamforming using all mics from both arrays to enhance audio from the estimated source position.
    
    Parameters:
    - dir_path: Directory containing WAV files
    - device1_index, device2_index: Indices of the two ReSpeaker devices
    - chunk_number: Chunk to process
    - source_position: (x, y) position of the source in meters
    - array_distance: Distance between the two arrays in meters
    
    Returns:
    - Enhanced audio signal
    - Sampling rate
    """
    # Find WAV files for both arrays
    mic_files1 = find_wav_files(dir_path, device1_index, chunk_number)
    mic_files2 = find_wav_files(dir_path, device2_index, chunk_number)
    
    print(f"Processing {len(mic_files1)} files from device {device1_index} and {len(mic_files2)} files from device {device2_index}")
    
    # Load all mic signals
    mic_signals = []
    fs = None
    
    for file_path in mic_files1 + mic_files2:
        sample_rate, signal = wavfile.read(file_path)
        if fs is None:
            fs = sample_rate
        elif fs != sample_rate:
            # Resample if rates differ
            ratio = fs / sample_rate
            resampled = resample(signal, int(len(signal) * ratio))
            mic_signals.append(resampled)
            continue
        
        mic_signals.append(signal)
    
    # Create microphone position arrays
    # First array at origin, second array at (array_distance, 0)
    mic_array = MicArray()
    
    # Combine mic positions from both arrays
    mic_positions = np.zeros((8, 2))
    # First array mics
    mic_positions[:4] = mic_array.mic_positions
    # Second array mics
    mic_positions[4:] = mic_array.mic_positions + np.array([array_distance, 0])
    
    # Perform location-based beamforming
    enhanced_signal = location_based_beamforming(mic_signals, fs, mic_positions, source_position)
    
    # Save enhanced signal
    output_file = f"{dir_path}/enhanced_chunk{chunk_number}.wav"
    wavfile.write(output_file, fs, enhanced_signal.astype(np.int16))
    print(f"Enhanced audio saved to {output_file}")
    
    return enhanced_signal, fs

# Main function would need to be updated to pass array_distance to beamform_from_all_mics function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python multi_array.py <directory> <device1_index> <device2_index> <chunk_number> [array_distance]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    device1_index = int(sys.argv[2])
    device2_index = int(sys.argv[3])
    chunk_number = int(sys.argv[4])
    array_distance = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0  # default 1 meter
    
    try:
        # Localize the sound source
        (x, y), (doa1, doa2), _ = localize_from_two_arrays(
            dir_path, device1_index, device2_index, chunk_number, array_distance, plot=True
        )
        
        # Perform beamforming using the exact source location
        source_position = np.array([x, y])
        enhanced_signal, fs = beamform_from_all_mics(
            dir_path, device1_index, device2_index, chunk_number, source_position, array_distance
        )
        
        print(f"Localization and beamforming complete.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()