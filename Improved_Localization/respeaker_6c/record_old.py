import usb.core
import usb.util
import pyaudio
import wave
import numpy as np
from tuning import Tuning
import json
import time
from datetime import datetime
import boto3
import os
import subprocess
from decimal import Decimal
from collections import defaultdict
import sys
import argparse

from edit_speaker import start_monitoring  # Import the monitoring function


RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
#RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
CHUNKSIZE = 15 # sec



def find_device(index):
    devices = [dev for dev in usb.core.find(find_all=True, idVendor=0x2886, idProduct=0x0018) 
           if any(usb.util.get_string(dev, intf.iInterface) == "SEEED Control" for cfg in dev for intf in cfg)]

    if not devices:
        raise Exception("No USB devices found.")
    if index >= len(devices):
        raise Exception(f"Device index {index} out of range. Only {len(devices)} devices found.")
    return devices[index]
    # previously was return devices[index], added below
    #device = devices[index]
    # Loop through the device configurations and pick the one that has 'SEEED Control' in its interface
    #for cfg in device:
    #    for intf in cfg:
    #        if 'SEEED Control' in usb.util.get_string(device, intf.iInterface):
    #            return device
    #raise Exception("Device with SEEED Control interface not found.")

def open_audio_stream(p, RESPEAKER_INDEX):
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
    )
    return stream


def record_audio(stream, p, dev, audio_file, doa_file, unknown_speakers, start_time):
    data_list = []
    count = 0

    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    for i in range(0, int(RESPEAKER_RATE / CHUNK * CHUNKSIZE)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        wf.writeframes(data)
        
        Mic_tuning = Tuning(dev)
        if count < CHUNKSIZE*10:
            if RESPEAKER_RATE / CHUNK / 10 * (count) < i < RESPEAKER_RATE / CHUNK / 10 * (count + 1):
                # Get DOA
                doa = Mic_tuning.direction
                timestamp = time.time()

                # If no match found, assign a new temporary ID
                if ID is None:
                    ID = f'temp_{len(unknown_speakers) + 1}'
                    unknown_speakers[ID]['start'] = min(unknown_speakers[ID]['start'], doa - 20)
                    unknown_speakers[ID]['end'] = max(unknown_speakers[ID]['end'], doa + 20)
                    unknown_speakers[ID]['ids'].append(ID)

                data_list.append({'doa': doa, 'timestamp': timestamp, 'record_time': timestamp - start_time, 'speaker': ID})
                print(str(timestamp - start_time) + ', ' + str(doa), " ", ID)
                count += 1
             
            with open (doa_file, 'w') as fj:
                json.dump(data_list,fj)

    return unknown_speakers

def close_audio_stream(stream, p):
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser(description="directory")
    parser.add_argument("-d", "--directory", required=True, help="directory that will contain the dataset")
    parser.add_argument("-s", "--second", required=True, help="recording duration")
    parser.add_argument("-i", "--index", required=True, help="Index of the ReSpeaker device")  # New argument for index
    parser.add_argument("-o", "--other_index", required=True, help="Index of the other device") 
    parser.add_argument("-t", "--start_time", required=True, help="Shared start time for all recordings")  # Add start_time argument
    args = parser.parse_args()
    dir_name = args.directory
    duration = args.second
    RESPEAKER_INDEX = int(args.index)  # Set the microphone index from the argument
    start_time = float(args.start_time)  # Shared start time

    dir_path = dir_name + '/recorded_data/'
    
    sec = int(duration)
    os.environ['LAST_ITERATION'] = str(sec)
    iteration = 0
    unknown_speakers = defaultdict(lambda: {'start': float('inf'), 'end': float('-inf'), 'ids': []})


    # Directory path for storing JSON files
    dir_path = 'data/recorded_data/'

    # Start recording
    while True:
        if iteration >= sec:
            print("DONE RECORDING")

            break
        else:
            dev = find_device(int(args.index) - 2)
            p = pyaudio.PyAudio()
            stream = open_audio_stream(p, RESPEAKER_INDEX)
            iteration += CHUNKSIZE
            audio_file = dir_path + f'chunk_{RESPEAKER_INDEX}_{iteration}.wav'
            doa_file   = dir_path + f'DOA_{RESPEAKER_INDEX}_{iteration}.json'
            

            print("RECORDING STARTED")
                
            unknown_speakers = record_audio(stream, p, dev, audio_file, doa_file, unknown_speakers, start_time)
            close_audio_stream(stream, p)

            print(str(iteration) + ' of '+ str(sec) + ' seconds are recorded')

if __name__ == '__main__':
    main()
