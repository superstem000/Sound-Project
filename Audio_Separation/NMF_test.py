import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import soundfile as sf

# Load audio file
file_path = "Test1.wav"  # Replace with your file path
y, sr = librosa.load(file_path, sr=None, mono=True)

# Compute the Short-Time Fourier Transform (STFT) and magnitude spectrogram
#       Fourier transform is performed on 'sliding windows' over the wav file to generate the spectogram 
#       array of amplitude --> 2d matrix of frequency specific amplitude
n_fft = 2048
hop_length = 512
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
magnitude, phase = librosa.magphase(D)  # Separate magnitude and phase

# Perform Non-Negative Matrix Factorization
#       lossfunction = kullback-leibler - diff between two probability distribution, often produce 'sparce' 
#       (regions of zero ampl, focus on smaller ampl)
#       mu (multiplicative update) - how to solve
#       nndvsda - how to initialize separation
n_components = 2  # Number of sound sources to separate
model = NMF(n_components=2, beta_loss='kullback-leibler', solver='mu', init='nndsvda', random_state=42)
W = model.fit_transform(magnitude.T)  # Basis matrix
H = model.components_  # Activation matrix

# Apply Wiener filtering for post-processing
#       so Wi @ Hi is reconstructed spectogram of i'th source (0 & 1)
#       np.dot(W, H) is the full reconstruction (approximation of original spectogram)
#       so each mask = contribution of each source in the full spectogram
#       ex) n time's m frequency --> 70% source1 & 30% source2 --> separated accordingly
masks = [(W[:, [i]] @ H[[i], :]) / np.dot(W, H) for i in range(n_components)]
masks = [mask.T for mask in masks]  # Transpose all masks

# Reconstruct separated sources
separated_sources = []
for i, mask in enumerate(masks):
    source_magnitude = mask * magnitude
    source_audio = librosa.istft(source_magnitude * phase, hop_length=hop_length)  # Combine with original phase
    separated_sources.append(source_audio)

    # Save separated audio files
    output_path = f"source2_{i+1}.wav"
    sf.write(output_path, source_audio, sr)
    print(f"Saved separated source {i+1} to {output_path}")

# Plot the original and separated spectrograms
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, hop_length=hop_length,
                         y_axis='log', x_axis='time')
plt.title("Original Spectrogram")
plt.colorbar(format="%+2.0f dB")

for i, mask in enumerate(masks):
    plt.subplot(1, 3, i+2)
    source_spectrogram = mask * magnitude
    librosa.display.specshow(librosa.amplitude_to_db(source_spectrogram, ref=np.max), sr=sr, hop_length=hop_length,
                             y_axis='log', x_axis='time')
    plt.title(f"Source {i+1} Spectrogram")
    plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.show()