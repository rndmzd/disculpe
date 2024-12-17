import pyaudio


def list_audio_devices():
    """
    Generator that yields detailed information about each available audio device.
    """
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            yield {
                "index": i,
                "name": device_info["name"],
                "max_input_channels": device_info["maxInputChannels"],
                "max_output_channels": device_info["maxOutputChannels"],
                "default_sample_rate": int(device_info["defaultSampleRate"]),
            }
    finally:
        p.terminate()


def display_audio_devices():
    print("Available Audio Devices:\n")
    for device in list_audio_devices():
        print(f"Device {device['index']}: {device['name']}")
        print(f"    Max Input Channels: {device['max_input_channels']}")
        print(f"    Max Output Channels: {device['max_output_channels']}")
        print(f"    Default Sample Rate: {device['default_sample_rate']} Hz\n")


if __name__ == "__main__":
    display_audio_devices()
