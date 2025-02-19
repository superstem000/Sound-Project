import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
import sounddevice as sd

def main():
    # Parameters for audio input
    sample_rate = 44100  # Sampling frequency in Hz
    block_size = 1 # How many samples do we group to feed to audio_callback
    duration = 10  # Total duration to capture audio in seconds
    threshold = 0.01  # Energy threshold to detect sound events
    min_event_duration = 0.05  # Minimum duration for an event in seconds

    sample_index = 0  # Keep track of current index in `audio_data`

    total_samples = duration * sample_rate
    audio_data = np.zeros(total_samples)  # Preallocate an array of zeros
    
    # Function to compute the energy of an audio frame
    def compute_energy(frame):
        return np.sqrt(np.mean(frame**2))

    # Callback function to process audio in real-time
    def audio_callback(indata, frames, time_info, status):
        nonlocal sample_index

        if status:
            print(status)  # Print any errors that occur in the audio stream
        
        # Store incoming audio data directly into preallocated array
        audio_data[sample_index:sample_index + frames] = indata[:, 0]  # Store channel 0 data
        sample_index += frames  # Increment the sample index

    # Start recording with the callback function
    print("Recording started...")
    
    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size, callback=audio_callback):
        sd.sleep(duration * 1000)

    print("Recording finished.")

    expected_samples = duration * sample_rate  # Should be 441,000 for 10 seconds at 44100Hz
    print(f"Expected samples: {expected_samples}, Actual samples: {sample_index}")

    # After recording: Post-processing for sound event detection
    event_times = []  # Store detected event start and end times
    buffer_size = 500  # Size of the buffer for averaging (matches the initial intent)
    energy_buffer = np.zeros(buffer_size)  # Fixed-size NumPy array for energy values
    buffer_index = 0  # Index to keep track of the current position in the buffer

    # Sliding window parameters
    window_size = 1  # Number of samples to compute energy for each step (should match block_size if needed)
    step_size = window_size  # Advance by the size of the window each time
    total_windows = len(audio_data) // step_size

    # Event tracking state variables
    is_in_event = False  # Whether we are currently inside a "sound event"
    event_start_time = None  # To track the start of a sound event

    # Iterate over audio data using a sliding window approach
    for window_idx in range(total_windows):
        # Calculate the current window's start and end index
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        # Ensure we do not go out of bounds
        if end_idx > len(audio_data):
            break

        # Compute energy for the current window
        window_data = audio_data[start_idx:end_idx]
        current_energy = compute_energy(window_data)
        #energy = compute_energy(window_data)

        # Update the energy buffer
        energy_buffer[buffer_index] = window_data
        buffer_index = (buffer_index + 1) % buffer_size  # Loop back to the start of the buffer

        # Compute smoothed energy from the buffer
        average_energy = compute_energy(energy_buffer)

        # Calculate the corresponding time for this window
        current_time = start_idx / sample_rate  # Time in seconds

        # Event detection logic using a state machine
        if current_energy > threshold:  # Sound detected above threshold
            if not is_in_event:  # New event starts
                event_start_time = current_time
                is_in_event = True  # Enter "IN_EVENT" state
                energy_buffer.fill(100)
        if average_energy * 2 <= threshold or current_time == total_samples / sample_rate:  # Below threshold
            if is_in_event:  # End of the current event
                event_duration = current_time - event_start_time
                if event_duration >= min_event_duration:  # Check minimum event duration
                    event_times.append((event_start_time, current_time))  # Store event times
                is_in_event = False  # Exit "IN_EVENT" state

    # Create a time axis for the audio data
    audio_time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

    # Calculate max and min values from the audio data for y-axis limits
    max_amplitude = np.max(audio_data)
    min_amplitude = np.min(audio_data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(audio_time_axis, audio_data, label='Audio Signal')
    plt.title('Audio Signal over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.ylim(min_amplitude * 1.1, max_amplitude * 1.1)  # Set limits based on audio data
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # Annotate event times on the plot
    for event_time, _ in event_times:
        plt.annotate(f'{event_time:.3f}', xy=(event_time, max_amplitude - 0.5 * max_amplitude), 
                     xytext=(event_time, max_amplitude - 0.45 * max_amplitude),
                     arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=7, color='black')
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('audio_signal_plot.png')
    plt.close()  # Close the figure
    
if __name__ == '__main__':
    main()
