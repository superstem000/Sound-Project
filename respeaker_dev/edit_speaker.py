import threading
import time
import json
import os
import re

# Temporary storage for unmatched entries (per array index)
unmatched_entries = {}

def monitor_and_process_json(dir_path, index, other_index):
    """Continuously checks for JSON pairs and processes them based on timestamp intervals."""
    processed_timestamps = set()

    if index not in unmatched_entries:
        unmatched_entries[index] = []
    if other_index not in unmatched_entries:
        unmatched_entries[other_index] = []

    while True:
        json_files = [f for f in os.listdir(dir_path) if f.startswith("DOA_") and f.endswith('.json')]
        timestamp_map = {}

        # Organize JSONs by timestamp
        for f in json_files:
            match = re.search(r'DOA_(\d+)_(\d+)\.json$', f)  # Extract (index, timestamp)
            if match:
                idx, timestamp = match.groups()
                if timestamp not in timestamp_map:
                    timestamp_map[timestamp] = {}
                timestamp_map[timestamp][idx] = f

        # Process files based on their timestamps
        for timestamp, files in sorted(timestamp_map.items()):
            if timestamp in processed_timestamps:
                continue  # Skip already processed timestamps

            # Get the paths to the JSON files for both indices
            json1_path = os.path.join(dir_path, files.get(index, ''))
            json2_path = os.path.join(dir_path, files.get(other_index, ''))

            # Only proceed if both files exist
            if not (os.path.exists(json1_path) and os.path.exists(json2_path)):
                continue  # If either file does not exist, skip this timestamp

            # Check if the current timestamp is a multiple of 15 (i.e., 15, 30, 45, 60, ...)
            timestamp_int = int(timestamp)
            if timestamp_int % 15 == 0:
                # Calculate the previous timestamp (i.e., for 30s -> 15s, 45s -> 30s, etc.)
                previous_timestamp = str(timestamp_int - 15)

                # If the previous 15-second files exist, process them
                if previous_timestamp in timestamp_map:
                    files_previous = timestamp_map[previous_timestamp]
                    json1_path_previous = os.path.join(dir_path, files_previous.get(index, ''))
                    json2_path_previous = os.path.join(dir_path, files_previous.get(other_index, ''))

                    if os.path.exists(json1_path_previous) and os.path.exists(json2_path_previous):
                        # Process the previous 15-second pair
                        unmatched_1, unmatched_2 = process_json_pair(json1_path_previous, json2_path_previous, index, other_index)
                        
                        # Store unmatched data for the next round
                        unmatched_entries[index].extend(unmatched_1)
                        unmatched_entries[other_index].extend(unmatched_2)

                # Mark this timestamp as processed
                processed_timestamps.add(timestamp)

        time.sleep(3)  # Check every 3 seconds




def process_json_pair(json1_path, json2_path, index, other_index):
    print("Processing")
    """Process JSON pair and return unmatched entries."""
    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    unmatched_1, unmatched_2 = [], []

    # Process matching DOAs and store unmatched ones
    for entry in data1[:]:  # Loop over a copy of data1
        closest_doa = find_closest_doa(data2, entry['record_time'])
        if closest_doa is None:
            print("None")
            unmatched_1.append(entry)
            data1.remove(entry)  # Remove from the original list
    
    for entry in data2[:]:  # Loop over a copy of data2
        closest_doa = find_closest_doa(data1, entry['record_time'])
        if closest_doa is None:
            print("None")
            unmatched_2.append(entry)
            data2.remove(entry)  # Remove from the original list

    # Send both JSONs for editing (ignoring function implementation for now)
    edit_json_pair(json1_path, json2_path, index, other_index)

    return unmatched_1, unmatched_2


def find_closest_doa(data, target_time):
    # Finds the 'doa' value in data with the closest 'record_time' to the target_time
    closest_entry = min(data, key=lambda x: abs(x['record_time'] - target_time))
    
    # Check if the time difference is greater than 0.15 seconds
    if abs(closest_entry['record_time'] - target_time) > 0.15:
        return None
    
    return closest_entry['doa']

def append_unmatched_data(json_path, index):
    """Append unmatched entries to the next JSON before processing."""
    if unmatched_entries[index]:
        with open(json_path, 'r+') as f:
            data = json.load(f)
            data.extend(unmatched_entries[index])  # Add stored unmatched values
            f.seek(0)
            json.dump(data, f, indent=4)

        unmatched_entries[index] = []  # Clear temporary storage

def find_closest_person(mic_data1, mic_data2, curr_doa, closest_doa):
    closest_person = None
    smallest_error = float('inf')  # Initialize with a very large error value

    # Loop through all persons
    for person in mic_data1:
        # Compare mic_data1 with curr_doa
        diff1 = abs(mic_data1[person]['doa'] - curr_doa)
        # Compare mic_data2 with closest_doa
        diff2 = abs(mic_data2[person]['doa'] - closest_doa)

        # Only consider the person if both errors are within 10 degrees
        if diff1 <= 10 and diff2 <= 10:
            total_error = diff1 + diff2  # Sum of both errors

            # If this person has a smaller total error, update the closest_person
            if total_error < smallest_error:
                smallest_error = total_error
                closest_person = person

    return closest_person


def edit_json_pair(json1_path, json2_path, index, other_index):
    """Edit JSON pair and assign speakers based on DOA match."""
    print("editing")
    # Load the JSON files at json1_path and json2_path
    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Load the mic{index}ID.json files
    mic_index_path = f'data/assign_speaker/mic{index}ID.json'
    mic_other_index_path = f'data/assign_speaker/mic{other_index}ID.json'

    with open(mic_index_path, 'r') as mic1, open(mic_other_index_path, 'r') as mic2:
        mic_data1 = json.load(mic1)
        mic_data2 = json.load(mic2)

    # Process the data in json1 (this one only)
    for entry in data1:
        # Find the corresponding DOA from json2 using the existing method
        closest_doa = find_closest_doa(data2, entry['record_time'])
        
        if closest_doa is not None:
            # Find the closest person based on the closest DOA
            closest_person = find_closest_person(mic_data1, mic_data2, entry['doa'], closest_doa)
            
            if closest_person:
                print(closest_person)
                entry['speaker'] = closest_person  # Assign the speaker based on the closest DOA


    # Write the updated data back to the JSON files
    with open(json1_path, 'w') as f1:
        json.dump(data1, f1, indent=4)

    print("write complete")




# To start the monitoring in a thread, use this function:
def start_monitoring(dir_path, index, other_index):
    print("monitoring...")
    monitor_thread = threading.Thread(target=monitor_and_process_json, args=(dir_path, index, other_index))
    monitor_thread.daemon = True
    monitor_thread.start()
