import pyaudio
import wave
import sys
import threading
import queue

# Import your transcription library here
# from faster_whisper import WhisperModel  # Uncomment if using faster-whisper


def list_audio_devices():
    """
    Generator that yields the index, name, max input channels, and max output channels
    of each available audio device.
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
            }
    finally:
        p.terminate()


def display_audio_devices():
    print("Available Audio Devices:\n")
    for device in list_audio_devices():
        print(f"Device {device['index']}: {device['name']}")
        print(f"    Max Input Channels: {device['max_input_channels']}")
        print(f"    Max Output Channels: {device['max_output_channels']}\n")


def get_device_index_by_name(target_name):
    """
    Returns the device index for a given device name.
    """
    p = pyaudio.PyAudio()
    try:
        for device in list_audio_devices():
            if device["name"] == target_name:
                return device["index"], device["max_input_channels"]
        return None, None
    finally:
        p.terminate()


def record_audio(device_name, duration=5, output_filename="output.wav"):
    """
    Records audio from the specified device for a given duration.
    """
    p = pyaudio.PyAudio()
    try:
        device_index, max_input_channels = get_device_index_by_name(device_name)
        if device_index is None:
            print(f"Device '{device_name}' not found.")
            sys.exit(1)

        # Define audio stream parameters
        FORMAT = pyaudio.paInt16  # 16-bit resolution
        CHANNELS = min(1, max_input_channels)  # Use 1 if supported, else max available
        RATE = 16000  # 16kHz sampling rate
        CHUNK = 1024  # Buffer size
        RECORD_SECONDS = duration
        WAVE_OUTPUT_FILENAME = output_filename

        # Open the stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
        )

        print(f"Recording from '{device_name}' for {RECORD_SECONDS} seconds...")

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording complete.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded data as a WAV file
        wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        print(f"Audio saved to '{WAVE_OUTPUT_FILENAME}'.")
    finally:
        p.terminate()


# Example usage for capturing audio
if __name__ == "__main__":
    # Display available devices
    display_audio_devices()

    # Specify the device name you want to record from
    desired_device_name = "VB-Audio Virtual Cable (VB-Audio Virtual Cable)"

    # Record audio
    record_audio(desired_device_name, duration=5, output_filename="test_recording.wav")
