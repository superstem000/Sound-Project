import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load audio files
mixed_file = 'mixed_noise.wav'     # Our mixed signal (like individual mic input)
ref_file = 'voice_noise.wav'       # Our reference signal (like beamformed output)

# Load with soundfile
y_mixed, sr = sf.read(mixed_file)
y_ref, _ = sf.read(ref_file)

# Ensure mono audio
if len(y_mixed.shape) > 1:
    y_mixed = y_mixed[:, 0]
if len(y_ref.shape) > 1:
    y_ref = y_ref[:, 0]

# Make sure both signals have the same length
min_len = min(len(y_mixed), len(y_ref))
y_mixed = y_mixed[:min_len]
y_ref = y_ref[:min_len]

print(f"Processing audio at {sr}Hz, length: {min_len/sr:.2f} seconds")

# STFT parameters
n_fft = 2048       # FFT size
hop_length = 512   # Hop size
win_length = 2048  # Window size

# Compute complex spectrograms
S_mixed = librosa.stft(y_mixed, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
S_ref = librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# Get magnitudes
mag_mixed = np.abs(S_mixed)
mag_ref = np.abs(S_ref)

# Keep the original phase for reconstruction
phase_mixed = np.angle(S_mixed)

# Wiener filter parameters
# Try different beta values based on your beamforming quality
# Higher beta = sharper separation but potentially more artifacts
# Lower beta = smoother separation but potentially more leakage
beta_values = {
    'low': 1.0,      # For poor beamforming (lots of interference)
    'medium': 2.0,   # Balanced option (default)
    'high': 4.0,     # For high-quality beamforming
    'very_high': 8.0 # For extremely precise beamforming
}

# Select beta value (change as needed)
beta = beta_values['high']  # Default to medium

# Estimate the noise component
# We can estimate this as difference between mixed and reference
noise_estimate = mag_mixed - mag_ref
noise_estimate = np.maximum(noise_estimate, 0)  # Ensure non-negative

# Small constant to avoid division by zero
eps = 1e-10

# Create Wiener-inspired mask with the selected beta
mask = (mag_ref**beta) / (mag_ref**beta + noise_estimate**beta + eps)

# Optional: Apply mask smoothing to reduce musical artifacts
# Adjust sigma values for different amounts of smoothing
# First value is time smoothing, second is frequency smoothing
time_smoothing = 0.5    # Higher = more smoothing over time
freq_smoothing = 1    # Higher = more smoothing over frequency
mask_smoothed = gaussian_filter(mask, sigma=(time_smoothing, freq_smoothing))

# Apply normalization to ensure masks sum to 1
voice_mask = mask_smoothed
music_mask = 1 - (mask_smoothed ** 0.1)  # The power < 1 makes the rejection stronger
mask_sum = voice_mask + music_mask  # Should theoretically be 1 already
voice_mask_normalized = voice_mask / mask_sum
music_mask_normalized = music_mask / mask_sum

# Apply normalized masks to the mixed spectrogram (preserving original phase)
S_target = S_mixed * voice_mask_normalized
S_noise = S_mixed * music_mask_normalized



# Inverse STFT to get time-domain signals
y_target = librosa.istft(S_target, hop_length=hop_length, win_length=win_length)
y_noise = librosa.istft(S_noise, hop_length=hop_length, win_length=win_length)

# Save results
sf.write(f'voice_wiener_beta{beta}.wav', y_target, sr)
sf.write(f'music_wiener_beta{beta}.wav', y_noise, sr)

# Optional: Create visualization
plt.figure(figsize=(12, 9))

# Plot spectrograms
plt.subplot(3, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(mag_mixed, ref=np.max), 
                        sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title('Mixed Signal Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(mag_ref, ref=np.max), 
                        sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title('Reference Signal Spectrogram')
plt.colorbar(format='%+2.0f dB')

# Plot mask
plt.subplot(3, 2, 3)
librosa.display.specshow(mask, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title(f'Wiener Mask (β={beta}, Unsmoothed)')
plt.colorbar()

plt.subplot(3, 2, 4)
librosa.display.specshow(mask_smoothed, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title(f'Wiener Mask (β={beta}, Smoothed)')
plt.colorbar()

# Plot separated spectrograms
plt.subplot(3, 2, 5)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_target), ref=np.max), 
                        sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title('Separated Voice')
plt.colorbar(format='%+2.0f dB')

plt.subplot(3, 2, 6)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_noise), ref=np.max), 
                        sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.title('Separated Music')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.savefig(f'wiener_separation_beta{beta}.png')
plt.close()

print(f"Separation complete with β={beta}")
print(f"Output files: voice_wiener_beta{beta}.wav, music_wiener_beta{beta}.wav")
print(f"Visualization saved as: wiener_separation_beta{beta}.png")
print("\nRecommended β values based on beamforming quality:")
for quality, value in beta_values.items():
    print(f"- {quality.replace('_', ' ').title()} quality: β = {value}")