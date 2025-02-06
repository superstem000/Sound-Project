import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")
    print(f"  Channels: {info['maxInputChannels']}")
    print(f"  Rate: {info['defaultSampleRate']}")

p.terminate()