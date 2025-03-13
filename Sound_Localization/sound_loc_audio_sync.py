import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import os
import re
import time
import glob
import threading
import soundfile as sf # type: ignore
import sounddevice as sd
from scipy.io import wavfile

def calculate_position(doa1, doa2, d):
    # Your existing position calculation function
    doa1_rad = np.radians(doa1 + 90)
    doa2_rad = np.radians(doa2 + 90)
    
    # Calculate slopes
    if doa1 == 0 or doa1 == 180:
        m1 = np.inf
    else:
        m1 = np.tan(doa1_rad)

    if doa2 == 0 or doa2 == 180:
        m2 = np.inf
    else:
        m2 = np.tan(doa2_rad)
    
    if m1 == np.inf and m2 == np.inf:
        return None

    if m1 == np.inf:
        intersection_x = d / 2
        intersection_y = m2 * (intersection_x + d / 2)
    elif m2 == np.inf:
        intersection_x = -d / 2
        intersection_y = m1 * (intersection_x - d / 2)
    elif m1 == m2:
        intersection_x = 0
        intersection_y = 0
    else:
        intersection_x = d/2 * (m2 + m1) / (-m2 + m1)
        intersection_y = m2 * (intersection_x + d / 2)

    return intersection_x, intersection_y

def find_closest_doa(data, target_time):
    closest_entry = min(data, key=lambda x: abs(x['record_time'] - target_time))
    return closest_entry['doa']

def track_localization(data_1, data_2, d):
    positions = []
    times = []
    
    # Sort combined data by record_time
    combined_data = sorted(data_1 + data_2, key=lambda x: x['record_time'])

    for entry in combined_data:
        current_time = entry['record_time']
        current_doa1 = find_closest_doa(data_1, current_time)
        current_doa2 = find_closest_doa(data_2, current_time)
        
        position = calculate_position(current_doa1, current_doa2, d)
        if position is not None:
            positions.append(position)
            times.append(current_time)
    
    return positions, times

class SoundDevicePlayer:
    """Handles audio playback for multiple WAV files in sequence using sounddevice"""
    def __init__(self, wav_files):
        # Sort WAV files by their timestamps
        self.wav_files = sorted(wav_files, key=self._extract_time_from_filename)
        self.current_file_index = 0
        self.is_playing = False
        self.playback_thread = None
        self.playback_speed = 1.0
        self.current_stream = None
        
        # Load audio file durations
        self.file_durations = []
        for wav_file in self.wav_files:
            try:
                with sf.SoundFile(wav_file) as f:
                    duration = len(f) / f.samplerate
                    self.file_durations.append(duration)
            except Exception as e:
                print(f"Could not read {wav_file}: {e}")
                self.file_durations.append(15.0)  # Default to 15 seconds if can't read
        
        print(f"Loaded {len(wav_files)} WAV files for playback")
        for i, file in enumerate(self.wav_files):
            print(f"  {i+1}: {os.path.basename(file)} (Duration: {self.file_durations[i]:.2f}s)")
            
        # Create an audio status text for display
        self.status_text = "Audio ready"

    def _extract_time_from_filename(self, filename):
        """Extract the time component from chunk_{index}_{time}.wav"""
        match = re.search(r'chunk_\d+_(\d+)\.wav', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
    
    def start(self):
        """Start audio playback in a separate thread"""
        if not self.is_playing:
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def stop(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.current_stream:
            self.current_stream.stop()
            self.current_stream.close()
            self.current_stream = None
            
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def set_playback_speed(self, speed):
        """Set the playback speed factor"""
        # Note: We'll simulate speed changes by adjusting wait times 
        # since sounddevice doesn't directly support speed changes
        self.playback_speed = speed
        
        # Stop and restart to apply the new speed (will affect wait timing)
        if self.is_playing:
            self.stop()
            self.start()
    
    def _playback_loop(self):
        """Main audio playback loop"""
        while self.is_playing and self.current_file_index < len(self.wav_files):
            current_file = self.wav_files[self.current_file_index]
            
            # Load and play the current file
            try:
                # Load audio file
                data, samplerate = sf.read(current_file)
                
                # Handle mono/stereo
                if len(data.shape) == 1:  # mono
                    channels = 1
                else:
                    channels = data.shape[1]
                
                # Define callback for sounddevice
                def callback(outdata, frames, time, status):
                    if status:
                        print(status)
                    
                    # Calculate the current position in the audio file
                    current_pos = callback.frame_count
                    
                    if current_pos + frames > len(data):
                        # We've reached the end of the file
                        remaining = len(data) - current_pos
                        if remaining > 0:
                            if channels == 1:
                                outdata[:remaining, 0] = data[current_pos:current_pos+remaining]
                                outdata[remaining:, 0] = 0
                            else:
                                outdata[:remaining] = data[current_pos:current_pos+remaining]
                                outdata[remaining:] = 0
                        else:
                            outdata.fill(0)
                        
                        # Signal we're done with this file
                        callback.done = True
                    else:
                        if channels == 1:
                            outdata[:, 0] = data[current_pos:current_pos+frames]
                        else:
                            outdata[:] = data[current_pos:current_pos+frames]
                        
                        callback.frame_count += frames
                
                # Initialize callback attributes
                callback.frame_count = 0
                callback.done = False
                
                # Start the stream
                self.current_stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=max(2, channels),  # Ensure at least stereo output
                    callback=callback
                )
                self.current_stream.start()
                
                filename = os.path.basename(current_file)
                self.status_text = f"Playing: {filename} (Speed: {self.playback_speed:.1f}x)"
                print(self.status_text)
                
                # Wait for the audio to finish or for playback to be stopped
                start_time = time.time()
                adjusted_duration = self.file_durations[self.current_file_index] / self.playback_speed
                
                while (not callback.done and 
                       time.time() - start_time < adjusted_duration and
                       self.is_playing):
                    time.sleep(0.1)
                
                # Clean up the stream
                if self.current_stream:
                    self.current_stream.stop()
                    self.current_stream.close()
                    self.current_stream = None
                
                # Move to the next file if we're still playing
                if self.is_playing:
                    self.current_file_index += 1
                
            except Exception as e:
                print(f"Error playing {current_file}: {e}")
                self.status_text = f"Error: {os.path.basename(current_file)}"
                self.current_file_index += 1
                time.sleep(1)  # Brief pause before trying next file
        
        # When finished with all files, reset
        if self.current_file_index >= len(self.wav_files):
            self.status_text = "Playback complete"
            print("Audio playback complete")
        
        self.current_file_index = 0
        self.is_playing = False

class TimeSyncedTracker:
    def __init__(self, positions, times, d, zone_centers=None, wav_files=None):
        self.positions = positions
        self.times = times
        self.d = d
        
        # Default zone centers if none provided
        if zone_centers is None:
            self.zone_centers = [np.array([-100, 100]), np.array([100, 100])]
        else:
            self.zone_centers = zone_centers
            
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Initialize plot elements
        self.trail_length = 20  # Number of previous positions to show
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3, label='Recent Path')
        self.current_pos, = self.ax.plot([], [], 'bo', markersize=10, label='Current Position')
        
        # Plot microphone positions
        self.ax.scatter([-d/2, d/2], [0, 0], color='red', label='Microphones')
        self.ax.text(-d/2, 0.1, 'Mic 1', color='red', ha='center')
        self.ax.text(d/2, 0.1, 'Mic 2', color='red', ha='center')
        
        # Calculate and plot trapezoids
        array_center = np.array([0, 0])
        self.trapezoids = []
        for i, zone_center in enumerate(self.zone_centers):
            trapezoid = self.calculate_trapezoid(zone_center, array_center)
            trap_plot, = self.ax.plot(
                [p[0] for p in np.vstack((trapezoid, trapezoid[0]))],
                [p[1] for p in np.vstack((trapezoid, trapezoid[0]))],
                'g-', alpha=0.3, label=f'Detection Zone {i+1}'
            )
            self.trapezoids.append((trapezoid, trap_plot))
        
        # Add timestamp display
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                   verticalalignment='top', fontsize=12)
        
        # Set up the plot
        self.ax.grid(True)
        self.ax.set_xlabel('X Position (cm)')
        self.ax.set_ylabel('Y Position (cm)')
        self.ax.set_title('Time-Synchronized Sound Source Tracking with Audio')
        
        # Calculate axis limits with some padding
        x_min, x_max = -600, 600
        y_min, y_max = -600, 600
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        self.ax.legend()

        # Time synchronization variables
        self.start_time = None
        self.current_frame = 0
        self.paused = False
        self.playback_speed = 1.0  # Can be adjusted
        
        # Add play/pause button
        self.play_pause_ax = plt.axes([0.85, 0.01, 0.1, 0.05])
        self.play_pause_button = plt.Button(self.play_pause_ax, 'Pause')
        self.play_pause_button.on_clicked(self.toggle_pause)
        
        # Add speed control buttons
        self.speed_up_ax = plt.axes([0.75, 0.01, 0.08, 0.05])
        self.speed_down_ax = plt.axes([0.65, 0.01, 0.08, 0.05])
        self.speed_up_button = plt.Button(self.speed_up_ax, 'Speed +')
        self.speed_down_button = plt.Button(self.speed_down_ax, 'Speed -')
        self.speed_up_button.on_clicked(self.speed_up)
        self.speed_down_button.on_clicked(self.speed_down)
        
        # Sound toggle button
        self.sound_ax = plt.axes([0.55, 0.01, 0.08, 0.05])
        self.sound_button = plt.Button(self.sound_ax, 'Sound On')
        self.sound_button.on_clicked(self.toggle_sound)
        
        # Audio player setup
        self.audio_player = None
        if wav_files and len(wav_files) > 0:
            self.audio_player = SoundDevicePlayer(wav_files)
            
        # Speed and audio status displays
        self.speed_text = self.ax.text(0.02, 0.93, f'Speed: {self.playback_speed:.1f}x', 
                                    transform=self.ax.transAxes, verticalalignment='top', fontsize=10)
        self.audio_text = self.ax.text(0.02, 0.88, 'Audio: Ready', 
                                    transform=self.ax.transAxes, verticalalignment='top', fontsize=10)

    def calculate_trapezoid(self, zone_center, array_center, front_width=90, back_width=150, height=150):
        # Calculate angle from array center to zone center
        direction = zone_center - array_center
        angle = np.arctan2(direction[1], direction[0]) - np.pi / 2
        
        # Define trapezoid with narrow side facing array center
        trapezoid_local = np.array([
            [-front_width/2, 0],        # Front left (narrow side)
            [front_width/2, 0],         # Front right (narrow side)
            [back_width/2, height],     # Back right (wide side)
            [-back_width/2, height]     # Back left (wide side)
        ])
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Center the trapezoid on (0,0) before rotation
        trapezoid_centered = trapezoid_local - np.array([0, height/2])
        
        # Rotate and translate to final position
        trapezoid_global = np.dot(trapezoid_centered, rotation_matrix.T) + zone_center
        
        return trapezoid_global

    def is_point_in_trapezoid(self, point, trapezoid):
        """Check if a point is inside a trapezoid using the ray casting algorithm"""
        n = len(trapezoid)
        inside = False
        
        for i in range(n):
            j = (i + 1) % n
            if ((trapezoid[i][1] > point[1]) != (trapezoid[j][1] > point[1]) and
                point[0] < (trapezoid[j][0] - trapezoid[i][0]) * 
                (point[1] - trapezoid[i][1]) / 
                (trapezoid[j][1] - trapezoid[i][1]) + trapezoid[i][0]):
                inside = not inside
                
        return inside
    
    def toggle_pause(self, event):
        self.paused = not self.paused
        self.play_pause_button.label.set_text('Play' if self.paused else 'Pause')
        
        # Also pause/resume audio
        if self.audio_player:
            if self.paused:
                self.audio_player.stop()
            else:
                self.audio_player.start()
        
    def speed_up(self, event):
        self.playback_speed = min(10.0, self.playback_speed * 1.5)
        self.speed_text.set_text(f'Speed: {self.playback_speed:.1f}x')
        
        # Update audio player speed too
        if self.audio_player:
            self.audio_player.set_playback_speed(self.playback_speed)
        
    def speed_down(self, event):
        self.playback_speed = max(0.1, self.playback_speed / 1.5)
        self.speed_text.set_text(f'Speed: {self.playback_speed:.1f}x')
        
        # Update audio player speed too
        if self.audio_player:
            self.audio_player.set_playback_speed(self.playback_speed)
            
    def toggle_sound(self, event):
        if not self.audio_player:
            return
            
        if self.audio_player.is_playing:
            self.audio_player.stop()
            self.sound_button.label.set_text('Sound On')
        else:
            self.audio_player.start()
            self.sound_button.label.set_text('Sound Off')

    def update(self, frame_time):
        # Initialize start time on first call
        if self.start_time is None:
            self.start_time = time.time()
            self.animation_start_time = self.times[0]
            
            # Start audio playback when animation starts
            if self.audio_player and not self.paused:
                self.audio_player.start()
                self.sound_button.label.set_text('Sound Off')
            
        # Calculate elapsed time based on real time, adjusted by playback speed
        if not self.paused:
            real_elapsed = (time.time() - self.start_time) * self.playback_speed
            current_time = self.animation_start_time + real_elapsed
            
            # Find the appropriate frame for the current time
            next_frame = self.current_frame
            while (next_frame < len(self.times) - 1 and 
                   self.times[next_frame + 1] <= current_time):
                next_frame += 1
                
            # Update current frame if needed
            if next_frame != self.current_frame:
                self.current_frame = next_frame
        
        # Get current frame index
        frame = self.current_frame
        
        # Update trail
        start_idx = max(0, frame - self.trail_length)
        x_trail = [pos[0] for pos in self.positions[start_idx:frame+1]]
        y_trail = [pos[1] for pos in self.positions[start_idx:frame+1]]
        self.trail.set_data(x_trail, y_trail)
        
        # Update current position
        if frame < len(self.positions):
            current_pos = self.positions[frame]
            self.current_pos.set_data([current_pos[0]], [current_pos[1]])
            
            # Update time display
            self.time_text.set_text(f'Time: {self.times[frame]:.2f}s')
            
            # Update audio status
            if self.audio_player:
                self.audio_text.set_text(f'Audio: {self.audio_player.status_text}')
            
            # Find trapezoids containing the point
            containing_trapezoids = []
            
            for i, (trapezoid, trap_plot) in enumerate(self.trapezoids):
                if self.is_point_in_trapezoid(current_pos, trapezoid):
                    # Calculate distance to zone center
                    distance = np.linalg.norm(np.array(current_pos) - self.zone_centers[i])
                    containing_trapezoids.append((i, distance, trap_plot))
            
            # Reset all trapezoids to default color
            for _, trap_plot in self.trapezoids:
                trap_plot.set_color('g')
                trap_plot.set_alpha(0.3)
            
            # If point is in any trapezoids, color only the closest one
            if containing_trapezoids:
                # Sort by distance
                containing_trapezoids.sort(key=lambda x: x[1])
                # Color only the closest trapezoid
                closest_trapezoid = containing_trapezoids[0]
                closest_trapezoid[2].set_color('r')
                closest_trapezoid[2].set_alpha(0.4)
        
        return (self.trail, self.current_pos, self.time_text, self.speed_text, self.audio_text,
                *[trap_plot for _, trap_plot in self.trapezoids])

def time_synced_animation_with_audio(positions, times, d, zone_centers=None, wav_files=None):
    """
    Create an animation that respects the actual record_time values and plays audio
    """
    tracker = TimeSyncedTracker(positions, times, d, zone_centers, wav_files)
    
    # Create animation that updates based on real time
    anim = FuncAnimation(
        tracker.fig, 
        tracker.update,
        interval=33,  # ~30 FPS for smooth animation
        blit=True,
        repeat=False,
        cache_frame_data=False  # Important for time-based animation
    )
    
    plt.show()
    
    # Clean up audio when done
    if tracker.audio_player:
        tracker.audio_player.stop()
    
    return anim

def find_wav_files(folder_path):
    """Find all WAV files matching chunk_{index}_{time}.wav pattern"""
    wav_pattern = os.path.join(folder_path, "chunk_*_*.wav")
    return glob.glob(wav_pattern)

def animate_with_real_timing_and_audio(data_1, data_2, d, wav_files=None):
    # Calculate positions and times
    positions, times = track_localization(data_1, data_2, d)
    
    # Get zone centers
    with open("mic2ID.json", 'r') as f:
        mic2_data = json.load(f)
    with open("mic3ID.json", 'r') as f:
        mic3_data = json.load(f)

    zone_centers = []
    
    # Match persons between the two mic readings by ID
    for person_key in mic2_data:
        person_id = mic2_data[person_key]["ID"]
        
        # Find matching person in mic3 data
        matching_person = next(
            (p for p in mic3_data.values() if p["ID"] == person_id),
            None
        )
        
        if matching_person:
            doa1 = mic2_data[person_key]["doa"]
            doa2 = matching_person["doa"]
            position = calculate_position(doa1, doa2, d)
            zone_centers.append(position)
    
    # Create the time-synced animation with audio
    return time_synced_animation_with_audio(positions, times, d, zone_centers, wav_files)

# Function to group files by identifiers
def group_files_by_identifier(folder_path):
    grouped_files = {"DOA_2": [], "DOA_3": []}
    
    for filename in os.listdir(folder_path):
        if re.search(r"DOA_2", filename):
            grouped_files["DOA_2"].append(os.path.join(folder_path, filename))
        elif re.search(r"DOA_3", filename):
            grouped_files["DOA_3"].append(os.path.join(folder_path, filename))
    
    return grouped_files

def load_and_combine_files(file_list):
    combined_data = []
    for file in file_list:
        with open(file) as f:
            combined_data.extend(json.load(f))
    return combined_data

def process_files_with_time_sync_and_audio(folder_path, d):
    """Process all files and create a time-synchronized animation with audio"""
    grouped_files = group_files_by_identifier(folder_path)
    
    data_1 = load_and_combine_files(grouped_files["DOA_2"])
    data_2 = load_and_combine_files(grouped_files["DOA_3"])
    
    # Find all WAV files matching the chunk pattern
    wav_files = find_wav_files(folder_path)
    
    return animate_with_real_timing_and_audio(data_1, data_2, d, wav_files)

# Main function to run the time-synchronized animation with audio
def main():
    distance_between_mics = 50  # Adjust as needed
    anim = process_files_with_time_sync_and_audio(".", distance_between_mics)

if __name__ == "__main__":
    main()