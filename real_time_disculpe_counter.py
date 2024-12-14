import pyaudio
import numpy as np
import threading
import queue
import sys
import time

# Import your transcription library here
from faster_whisper import WhisperModel
# If using Vosk, import necessary modules
# from vosk import Model, KaldiRecognizer


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
    Returns the device index and max input channels for a given device name.
    """
    p = pyaudio.PyAudio()
    try:
        for device in list_audio_devices():
            if device["name"] == target_name:
                return device["index"], device["max_input_channels"]
        return None, None
    finally:
        p.terminate()


class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
        self.counter_file = "counter.txt"
        self.update_counter_file()

    def increment(self, num=1):
        with self.lock:
            self.count += num
            self.update_counter_file()

    def update_counter_file(self):
        with open(self.counter_file, "w", encoding="utf-8") as f:
            f.write(str(self.count))


def audio_callback(in_data, frame_count, time_info, status):
    """
    This function will be called for each audio block.
    """
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


def transcribe_audio(model, audio_queue, counter):
    """
    Continuously processes audio data from the queue and detects the keyword.
    """
    while True:
        try:
            data = audio_queue.get()
            # Convert byte data to numpy array
            audio_data = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            )  # Normalize
            # If using faster-whisper, process the audio_data
            segments, info = model.transcribe(audio_data, language='es', beam_size=5)
            text = " ".join([segment.text for segment in segments]).lower()

            # Placeholder for transcription
            # Replace the following line with actual transcription
            # text = mock_transcription(audio_data)

            if "disculpe" in text:
                occurrences = text.count("disculpe")
                counter.increment(occurrences)
                print(
                    f"Detected 'disculpe' {occurrences} time(s). Total: {counter.count}"
                )
        except Exception as e:
            print(f"Error in transcription thread: {e}")


def mock_transcription(audio_data):
    """
    Mock transcription function for demonstration.
    Replace this with actual transcription logic using faster-whisper or another library.
    """
    # For demonstration, we'll pretend "disculpe" is detected every 10 seconds
    current_time = time.time()
    if int(current_time) % 10 == 0:
        return "disculpe"
    return ""


if __name__ == "__main__":
    # Display available devices
    display_audio_devices()

    # Specify the device name you want to record from
    desired_device_name = (
        "CABLE Output (VB-Audio Virtual Cable)"  # Replace with your device's exact name
    )

    # Get device index and max input channels
    device_index, max_input_channels = get_device_index_by_name(desired_device_name)
    if device_index is None:
        print(f"Device '{desired_device_name}' not found.")
        sys.exit(1)

    # Define audio stream parameters
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1  # Mono
    RATE = 16000  # 16kHz sampling rate
    CHUNK = 1024  # Buffer size

    # Ensure the device supports the number of channels
    if max_input_channels < CHANNELS:
        print(f"Device '{desired_device_name}' does not support {CHANNELS} channel(s).")
        print(f"Setting CHANNELS to {max_input_channels}.")
        CHANNELS = max_input_channels

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Initialize audio queue and counter
    audio_queue = queue.Queue()
    counter = Counter()

    # Initialize transcription model
    # model = WhisperModel("small", device="cuda", compute_type="float16")  # Uncomment if using faster-whisper

    # Open the audio stream
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
            stream_callback=audio_callback,
        )

        print(f"Recording from '{desired_device_name}' with {CHANNELS} channel(s)...")
        stream.start_stream()

        # Start transcription thread
        transcription_thread = threading.Thread(
            target=transcribe_audio, args=(None, audio_queue, counter)
        )
        transcription_thread.daemon = True
        transcription_thread.start()

        print("Listening... Press Ctrl+C to stop.")

        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(0)
