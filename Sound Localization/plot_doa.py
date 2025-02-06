import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [{'doa': d['doa'], 'record_time': d['record_time']} for d in data]

def update(frame):
    ax.clear()
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('DOA over Time')
    
    times = [d['record_time'] for d in data[:frame+1]]
    doas = [np.radians(d['doa']) for d in data[:frame+1]]
    
    # Normalize times to go from 0 to 1
    normalized_times = [(t - min_time) / (max_time - min_time) for t in times]
    
    # Create a color gradient
    colors = plt.cm.viridis(normalized_times)
    
    # Plot the path
    ax.plot(doas, normalized_times, '-', color='red', alpha=0.5)
    
    # Plot each point
    for i in range(len(times)):
        ax.plot(doas[i], normalized_times[i], 'o', color=colors[i], markersize=8)
    
    # Highlight the current point
    if doas:
        ax.plot(doas[-1], normalized_times[-1], 'o', color='red', markersize=12, markerfacecolor='none')
    
    # Add a text annotation for the current time
    if times:
        ax.text(0.5, 0.5, f'Time: {times[-1]:.2f}s', transform=ax.transAxes, ha='center', va='center')

# Read the JSON file
file_path = 'DOA_3_15.json'  # Replace with your file path
data = read_json_file(file_path)

# Sort data by record_time
data.sort(key=lambda x: x['record_time'])

# Get min and max times for normalization
min_time = min(d['record_time'] for d in data)
max_time = max(d['record_time'] for d in data)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Create the animation
anim = FuncAnimation(fig, update, frames=len(data), interval=50, repeat=False)

plt.show()