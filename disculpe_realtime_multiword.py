import logging
import configparser
import pyaudio
import threading
import queue
import sys
import time
import json

from vosk import Model, KaldiRecognizer

log_file = "log.txt"
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config = configparser.ConfigParser()
config.read("config.ini")

counters_file = "counters.json"

with open(counters_file, "r") as f:
    counter_info = json.load(f)


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
    def __init__(self, counter_file):
        self.count = 0
        self.lock = threading.Lock()
        self.counter_file = counter_file
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


def transcribe_audio(model, audio_queue, sample_rate, target_words):
    """
    Continuously processes audio data from the queue and detects the keyword.
    Uses Vosk's KaldiRecognizer for transcription.
    """
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(False)  # Disable word-level timestamps for efficiency

    while True:
        try:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "").lower()

                for target_word in target_words:
                    if target_word in text:
                        occurrences = text.count(target_word)
                        counter_info[target_word]['counter'].increment(occurrences)
                        print(
                            f"Detected '{target_word}' {occurrences} time(s). Total: {counter_info[target_word]['counter'].count}"
                        )
            else:
                # For partial results, you can handle them if needed
                pass
        except Exception as e:
            print(f"Error in transcription thread: {e}")


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

    # Initialize audio queue
    audio_queue = queue.Queue()

    counter_models = {}
    for word in counter_info:
        if counter_info[word]['model'] not in counter_models:
            counter_models[counter_info[word]['model']] = []
        counter_models[counter_info[word]['model']].append(word)

    for target_word in counter_info:
        counter_info[target_word]['counter'] = Counter(counter_file=counter_info[target_word]['counter_file'])

    models = {}
    for model_path in counter_models:
        # Initialize Vosk model
        try:
            models[model_path] = Model(model_path)
        except Exception as e:
            print(f"Failed to load Vosk model from '{model_path}': {e}")
            sys.exit(1)

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

        # def transcribe_audio(model, audio_queue, counter, sample_rate, target_words):

        for model in models:
            # Start transcription thread
            transcription_thread = threading.Thread(
                target=transcribe_audio, args=(models[model], audio_queue, RATE, counter_models[model])
            )
            transcription_thread.daemon = True
            print(f"Starting transcription thread for model '{model}'")
            transcription_thread.start()

        print("Listening... Press Ctrl+C to stop.")

        while stream.is_active():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(0)
