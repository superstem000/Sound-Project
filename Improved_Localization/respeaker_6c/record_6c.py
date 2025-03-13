import usb.core
import usb.util
import pyaudio
import wave
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6  # Changed to 6 channels
RESPEAKER_WIDTH = 2
CHUNK = 1024
CHUNKSIZE = 15  # seconds of recording per chunk

def find_device(index):
    devices = [dev for dev in usb.core.find(find_all=True, idVendor=0x2886, idProduct=0x0018) 
           if any(usb.util.get_string(dev, intf.iInterface) == "SEEED Control" for cfg in dev for intf in cfg)]

    if not devices:
        raise Exception("No USB devices found.")
    if index >= len(devices):
        raise Exception(f"Device index {index} out of range. Only {len(devices)} devices found.")
    return devices[index]

def open_audio_stream(p, RESPEAKER_INDEX):
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
    )
    return stream

def record_audio(stream, p, dir_path, device_index, iteration, start_time):
    print(f"Recording {CHUNKSIZE} seconds from device {device_index}")
    
    # Record the exact start time of this chunk
    chunk_start_time = time.time()
    timestamp_data = {
        "device_index": device_index,
        "chunk_number": iteration,
        "start_time": chunk_start_time,
        "relative_time": chunk_start_time - start_time,
        "files": []
    }
    
    # Record audio
    frames = []
    for i in range(0, int(RESPEAKER_RATE / CHUNK * CHUNKSIZE)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    # Convert byte data to numpy array for easier channel separation
    data_array = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # Reshape the array so that each row is a single sample containing all channels
    samples = data_array.reshape(-1, RESPEAKER_CHANNELS)
    
    # Extract and save each channel separately
    for channel in range(RESPEAKER_CHANNELS):
        # Extract the channel data
        channel_data = samples[:, channel].tobytes()
        
        # Create a descriptive filename
        if channel == 0:
            channel_type = "asr"
        elif 1 <= channel <= 4:
            channel_type = f"mic{channel}"
        else:
            channel_type = "playback"
            
        channel_filename = f"device{device_index}_{channel_type}_chunk{iteration}.wav"
        full_path = os.path.join(dir_path, channel_filename)
        
        # Save the channel to a wave file
        wf = wave.open(full_path, 'wb')
        wf.setnchannels(1)  # Each file has only 1 channel
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
        wf.setframerate(RESPEAKER_RATE)
        wf.writeframes(channel_data)
        wf.close()
        
        # Add file info to timestamp data
        timestamp_data["files"].append({
            "filename": channel_filename,
            "channel": channel,
            "channel_type": channel_type
        })
        
        print(f"Saved channel {channel} to {full_path}")
    
    # Save timestamp data to JSON
    timestamp_filename = f"{dir_path}/device{device_index}_chunk{iteration}_timestamps.json"
    with open(timestamp_filename, 'w') as f:
        json.dump(timestamp_data, f, indent=2)
    
    print(f"Saved timestamp data to {timestamp_filename}")
    
    return chunk_start_time

def close_audio_stream(stream, p):
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    parser = argparse.ArgumentParser(description="Record audio from ReSpeaker devices")
    parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
    parser.add_argument("-s", "--second", required=True, help="recording duration in seconds")
    parser.add_argument("-i", "--index", required=True, help="Index of the ReSpeaker device")
    parser.add_argument("-o", "--other_index", required=True, help="Index of the other device")
    parser.add_argument("-t", "--start_time", required=True, help="Shared start time for all recordings")
    args = parser.parse_args()
    
    dir_name = args.directory
    total_duration = int(args.second)
    RESPEAKER_INDEX = int(args.index)
    start_time = float(args.start_time)
    
    # Create directory if it doesn't exist
    dir_path = os.path.join(dir_name, 'recorded_data')
    os.makedirs(dir_path, exist_ok=True)
    
    # Create a master timestamps file for this device
    master_timestamps = {
        "device_index": RESPEAKER_INDEX,
        "global_start_time": start_time,
        "chunks": []
    }
    
    # Calculate current time and wait if needed to sync with start_time
    current_time = time.time()
    if current_time < start_time:
        wait_time = start_time - current_time
        print(f"Waiting {wait_time:.2f} seconds to synchronize recording start...")
        time.sleep(wait_time)
    
    iteration = 0
    recorded_duration = 0
    
    # Start recording in chunks
    while recorded_duration < total_duration:
        try:
            # Find the ReSpeaker device
            dev = find_device(int(args.index) - 2)
            
            # Initialize PyAudio and open stream
            p = pyaudio.PyAudio()
            stream = open_audio_stream(p, RESPEAKER_INDEX)
            
            # Update iteration counter
            iteration += 1
            chunk_size = min(CHUNKSIZE, total_duration - recorded_duration)
            
            # Record audio, save channels, and get chunk start time
            chunk_start_time = record_audio(stream, p, dir_path, RESPEAKER_INDEX, iteration, start_time)
            
            # Add chunk info to master timestamps
            master_timestamps["chunks"].append({
                "chunk_number": iteration,
                "start_time": chunk_start_time,
                "relative_time": chunk_start_time - start_time,
                "duration": CHUNKSIZE
            })
            
            # Clean up
            close_audio_stream(stream, p)
            
            # Update recorded duration
            recorded_duration += CHUNKSIZE
            print(f'{recorded_duration} of {total_duration} seconds recorded')
            
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            break
    
    # Save master timestamps file
    master_timestamp_file = f"{dir_path}/device{RESPEAKER_INDEX}_master_timestamps.json"
    with open(master_timestamp_file, 'w') as f:
        json.dump(master_timestamps, f, indent=2)
    
    print(f"Saved master timestamps to {master_timestamp_file}")
    print("RECORDING COMPLETE")

if __name__ == '__main__':
    main()