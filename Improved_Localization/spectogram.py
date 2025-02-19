import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def plot_spectrogram(wav_file, fmin=0, fmax=5000):
    # Read the wav file
    fs, data = wavfile.read(wav_file)
    
    # If stereo, convert to mono
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Compute the spectrogram
    f, t, Sxx = spectrogram(data, fs)

    # Mask out frequencies outside the specified range (fmin, fmax)
    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.log10(Sxx), shading='auto')  # log scale for better visualization
    plt.title("Spectrogram of " + wav_file)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Log Power Spectral Density [dB]")
    plt.show()

# Example usage
plot_spectrogram("mic_4.wav")
