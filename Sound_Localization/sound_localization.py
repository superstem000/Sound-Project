import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import deque



def calculate_position(doa1, doa2, d):
    doa1_rad = np.radians(doa1 + 90)
    doa2_rad = np.radians(doa2 + 90)
    
    # Calculate slopes
    if doa1 == 0 or doa1 == 180:  # Vertical lines
        m1 = np.inf  # Undefined slope
    else:
        m1 = np.tan(doa1_rad)

    if doa2 == 0 or doa2 == 180:  # Vertical lines
        m2 = np.inf  # Undefined slope
    else:
        m2 = np.tan(doa2_rad)
    
    
    if m1 == np.inf and m2 == np.inf:
        return None  # Parallel lines (no intersection)

    
    if m1 == np.inf:  # Line 1 is vertical
        intersection_x = d / 2
        intersection_y = m2 * (intersection_x + d / 2)
    elif m2 == np.inf:  # Line 2 is vertical
        intersection_x = -d / 2
        intersection_y = m1 * (intersection_x - d / 2)
    else:
        intersection_x = d/2 * (m2 + m1) / (-m2 + m1)
        intersection_y = m2 * (intersection_x + d / 2)

    return intersection_x, intersection_y

def find_closest_doa(data, target_time):
    # Finds the 'doa' value in data with the closest 'record_time' to the target_time
    closest_entry = min(data, key=lambda x: abs(x['record_time'] - target_time))
    return closest_entry['doa']

def detect_events(data_1, data_2):
    events = []
    
    stable_doa1 = None
    stable_doa2 = None
    event_start_time = None
    threshold_duration = 0.5  # seconds
    first_event_detected = False  # Flag to track if the first event has been detected
    
    # Merge the datasets, assuming they are sorted by record_time
    combined_data = sorted(data_1 + data_2, key=lambda x: x['record_time'])

    for entry in combined_data:
        current_time = entry['record_time']
        current_doa1 = find_closest_doa(data_1, current_time)
        current_doa2 = find_closest_doa(data_2, current_time)
        
        # Initialize stable DOA values
        if stable_doa1 is None:
            stable_doa1 = current_doa1
            stable_doa2 = current_doa2
            event_start_time = current_time
        
        # Check for changes in DOA values
        if current_doa1 != stable_doa1 or current_doa2 != stable_doa2:
            # Update the stable DOAs if they have changed
            stable_doa1 = current_doa1
            stable_doa2 = current_doa2
            event_start_time = current_time  # Reset the timer
            first_event_detected = True
            
        # Check if both DOAs are stable for 0.5 seconds
        elif current_time - event_start_time >= threshold_duration:
            # Record the event with stable DOAs
            # Record the event with stable DOAs if it's not the initial DOA values
            if not first_event_detected and (stable_doa1 == data_1[0]['doa'] and stable_doa2 == data_2[0]['doa']):
                # Skip recording the initial stable DOAs
                continue
            
            
            if not any(event['doa1'] == current_doa1 and event['doa2'] == current_doa2 for event in events):
                    events.append({
                        'doa1': current_doa1,
                        'doa2': current_doa2,
                        'time': event_start_time
                    })
            # Reset the stable DOAs and start time for future events
            #stable_doa1 = None
            #stable_doa2 = None
            
    return events

# Plot the sound events and microphone arrays
def plot_events(events, d):
    plt.figure(figsize=(8, 8))
    
    # Plot microphone positions
    plt.scatter([-d/2, d/2], [0, 0], color='red', label='Microphones')
    plt.text(-d/2, 0.1, 'Mic 1', color='red', ha='center')
    plt.text(d/2, 0.1, 'Mic 2', color='red', ha='center')

    # Initialize the orientation using the first two events
    #initial_doa2 = int(events[1]['doa2'])
    initial_doa2 = 0
    initial_doa1 = 0
    #initial_doa1 = int(events[2]['doa1'])
    print(f"doa1: {initial_doa1} doa2: {initial_doa2}")
    
    # Plot sound events
    for i in range(0, len(events)):  # Start from index 3
        # Calculate the adjustments for DOA1 and DOA2
        event = events[i]
        adjusted_doa1 = (int(event['doa1']) - initial_doa1)
        adjusted_doa2 = (int(event['doa2']) - initial_doa2)

        # Normalize the adjusted_doa to the range of 0 to 360 degrees
        if adjusted_doa1 < 0:
            adjusted_doa1 += 360
        if adjusted_doa2 < 0:
            adjusted_doa2 += 360

        x, y = calculate_position(adjusted_doa2, adjusted_doa1, d)
        print(f"x position: {x} y position: {y} time: {event['time']}")
        plt.scatter(x, y, label=f'Event {i+1}')
        plt.text(x, y, f'Event {i+1}\n({event["time"]:.2f}s)', fontsize=9, ha='right')
    
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Sound Events')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to process the data and display the graph
def process_and_plot(data_1, data_2, d):
    events = detect_events(data_1, data_2)
    print(events)
    
    # Plot the events
    plot_events(events, d)

# Function to group files by identifiers
def group_files_by_identifier(folder_path):
    # Dictionary to store files grouped by identifier
    grouped_files = {"DOA_2": [], "DOA_3": []}

    # Scan folder for files
    for filename in os.listdir(folder_path):
        if re.search(r"DOA_2", filename):
            grouped_files["DOA_2"].append(os.path.join(folder_path, filename))
        elif re.search(r"DOA_3", filename):
            grouped_files["DOA_3"].append(os.path.join(folder_path, filename))

    return grouped_files

# Load and combine data from all files in a list
def load_and_combine_files(file_list):
    combined_data = []
    for file in file_list:
        with open(file) as f:
            combined_data.extend(json.load(f))
    return combined_data

# Process all combined data for DOA_2 and DOA_3 groups
def process_all_files(folder_path, d):
    grouped_files = group_files_by_identifier(folder_path)
    
    # Load and combine all files for DOA_2 and DOA_3
    data_1 = load_and_combine_files(grouped_files["DOA_2"])
    data_2 = load_and_combine_files(grouped_files["DOA_3"])
    
    # Call process_and_plot with the combined data
    process_and_plot(data_1, data_2, d)


distance_between_mics = 64  # Adjust as needed
process_all_files(".", distance_between_mics)