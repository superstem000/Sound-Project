import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import matplotlib.animation as animation

def calculate_position(doa1, doa2, d):
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

class LiveTracker:
    def __init__(self, positions, d):
        self.positions = positions
        self.d = d
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Initialize plot elements
        self.trail_length = 20  # Number of previous positions to show
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3, label='Recent Path')
        self.current_pos, = self.ax.plot([], [], 'bo', markersize=10, label='Current Position')
        
        # Plot microphone positions
        self.ax.scatter([-d/2, d/2], [0, 0], color='red', label='Microphones')
        self.ax.text(-d/2, 0.1, 'Mic 1', color='red', ha='center')
        self.ax.text(d/2, 0.1, 'Mic 2', color='red', ha='center')
        
        # Set up the plot
        self.ax.grid(True)
        self.ax.set_xlabel('X Position (cm)')
        self.ax.set_ylabel('Y Position (cm)')
        self.ax.set_title('Live Sound Source Tracking')
        
        # Calculate axis limits with some padding
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        padding = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.1
        
        self.ax.set_xlim(x_min - padding, x_max + padding)
        self.ax.set_ylim(y_min - padding, y_max + padding)
        
        self.ax.legend()

    def update(self, frame):
        # Update trail
        start_idx = max(0, frame - self.trail_length)
        x_trail = [pos[0] for pos in self.positions[start_idx:frame+1]]
        y_trail = [pos[1] for pos in self.positions[start_idx:frame+1]]
        self.trail.set_data(x_trail, y_trail)
        
        # Update current position
        if frame < len(self.positions):
            self.current_pos.set_data([self.positions[frame][0]], [self.positions[frame][1]])
        
        return self.trail, self.current_pos

def animate_trajectory(positions, d):
    tracker = LiveTracker(positions, d)
    
    # Create animation
    anim = FuncAnimation(tracker.fig, tracker.update, 
                        frames=len(positions),
                        interval=50,
                        blit=True,
                        repeat=False)  # Set repeat to False to prevent looping
    
    plt.show()
    
    return anim

# Main function to process the data and display the animation
def process_and_animate(data_1, data_2, d):
    positions, times = track_localization(data_1, data_2, d)
    return animate_trajectory(positions, d)

# Function to group files by identifiers
def group_files_by_identifier(folder_path):
    grouped_files = {"DOA_2": [], "DOA_1": []}
    
    for filename in os.listdir(folder_path):
        if re.search(r"DOA_2", filename):
            grouped_files["DOA_2"].append(os.path.join(folder_path, filename))
        elif re.search(r"DOA_1", filename):
            grouped_files["DOA_1"].append(os.path.join(folder_path, filename))
    
    return grouped_files

def load_and_combine_files(file_list):
    combined_data = []
    for file in file_list:
        with open(file) as f:
            combined_data.extend(json.load(f))
    return combined_data

def process_all_files(folder_path, d):
    grouped_files = group_files_by_identifier(folder_path)
    
    data_1 = load_and_combine_files(grouped_files["DOA_1"])
    data_2 = load_and_combine_files(grouped_files["DOA_2"])
    
    return process_and_animate(data_1, data_2, d)

# Run the animation
distance_between_mics = 64  # Adjust as needed
anim = process_all_files(".", distance_between_mics)