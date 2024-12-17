import pyaudio
import numpy as np
import threading
import queue
import sys
import time
import logging
from faster_whisper import WhisperModel  # Ensure faster-whisper is installed and compatible

# ---------------------------- Logging Configuration ----------------------------

# Configure logging to output DEBUG and higher level messages to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler("real_time_disculpe_counter.log", mode='w')  # Log to file
    ]
)

# ---------------------------- Helper Functions ----------------------------

def list_audio_devices():
    """
    Generator that yields detailed information about each available audio device.
    """
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            yield {
                'index': i,
                'name': device_info['name'],
                'max_input_channels': device_info['maxInputChannels'],
                'max_output_channels': device_info['maxOutputChannels'],
                'default_sample_rate': int(device_info['defaultSampleRate']),
            }
    finally:
        p.terminate()

def display_audio_devices():
    """
    Prints all available audio devices with their details.
    """
    logging.debug("Listing all available audio devices.")
    print("Available Audio Devices:\n")
    for device in list_audio_devices():
        print(f"Device {device['index']}: {device['name']}")
        print(f"    Max Input Channels: {device['max_input_channels']}")
        print(f"    Max Output Channels: {device['max_output_channels']}")
        print(f"    Default Sample Rate: {device['default_sample_rate']} Hz\n")
        logging.debug(f"Device {device['index']}: {device['name']}, "
                      f"Max Input Channels: {device['max_input_channels']}, "
                      f"Max Output Channels: {device['max_output_channels']}, "
                      f"Default Sample Rate: {device['default_sample_rate']} Hz")

def get_device_info_by_name(target_name):
    """
    Returns the device index, max input channels, and default sample rate for a given device name.
    """
    p = pyaudio.PyAudio()
    try:
        for device in list_audio_devices():
            if device['name'] == target_name:
                logging.debug(f"Found target device '{target_name}' with index {device['index']}.")
                return device['index'], device['max_input_channels'], device['default_sample_rate']
        logging.error(f"Device '{target_name}' not found among available devices.")
        return None, None, None
    finally:
        p.terminate()

class Counter:
    """
    Thread-safe counter to track the number of times 'disculpe' is detected.
    """
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
        self.counter_file = "counter.txt"
        self.update_counter_file()
        logging.debug("Counter initialized.")

    def increment(self, num=1):
        with self.lock:
            self.count += num
            logging.debug(f"Counter incremented by {num}. New count: {self.count}")
            self.update_counter_file()

    def update_counter_file(self):
        try:
            with open(self.counter_file, "w", encoding="utf-8") as f:
                f.write(str(self.count))
            logging.debug(f"Counter file '{self.counter_file}' updated with count: {self.count}")
        except Exception as e:
            logging.error(f"Failed to update counter file: {e}")

# ---------------------------- Audio Processing Functions ----------------------------

def audio_callback(in_data, frame_count, time_info, status):
    """
    This function will be called for each audio block.
    """
    if status:
        logging.warning(f"Stream status: {status}")
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def transcribe_audio(model, audio_queue, counter):
    """
    Continuously processes audio data from the queue and detects the keyword.
    """
    buffer = np.array([], dtype=np.float32)
    buffer_duration = 3  # seconds
    target_duration = 1  # seconds per chunk for transcription

    logging.debug("Transcription thread started.")

    while True:
        try:
            data = audio_queue.get()
            logging.debug("Received audio data from queue.")
            # Convert byte data to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
            buffer = np.concatenate((buffer, audio_data))
            logging.debug(f"Buffer size: {len(buffer)} samples.")

            # Check if buffer has enough data for target_duration
            if len(buffer) >= model.sample_rate * target_duration:
                # Extract chunk for transcription
                chunk = buffer[:model.sample_rate * target_duration]
                buffer = buffer[model.sample_rate * target_duration:]
                logging.debug(f"Processing chunk of {len(chunk)} samples.")

                # Transcribe
                segments, info = model.transcribe(chunk, language='en', beam_size=5)
                text = " ".join([segment.text for segment in segments]).lower()
                logging.debug(f"Transcribed Text: '{text}'")

                # Check for keyword
                if "disculpe" in text:
                    occurrences = text.count("disculpe")
                    counter.increment(occurrences)
                    logging.info(f"Detected 'disculpe' {occurrences} time(s). Total: {counter.count}")
        except Exception as e:
            logging.exception(f"Error in transcription thread: {e}")

# ---------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    # Display available devices
    display_audio_devices()

    # Specify the device name you want to record from
    desired_device_name = "CABLE Output (VB-Audio Virtual Cable)"  # Updated to use CABLE Output

    # Get device index, max input channels, and default sample rate
    device_index, max_input_channels, default_sample_rate = get_device_info_by_name(desired_device_name)
    if device_index is None:
        logging.error(f"Exiting script as device '{desired_device_name}' was not found.")
        sys.exit(1)

    # Define audio stream parameters based on device capabilities
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    if max_input_channels >= 2:
        CHANNELS = 2  # Stereo
    elif max_input_channels == 1:
        CHANNELS = 1  # Mono
    else:
        logging.error(f"Device '{desired_device_name}' does not support input channels.")
        sys.exit(1)
    RATE = default_sample_rate  # Use device's default sample rate
    CHUNK = 1024  # Buffer size

    logging.debug(f"Stream parameters set: FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Initialize audio queue and counter
    audio_queue = queue.Queue()
    counter = Counter()

    # Initialize Whisper model
    try:
        # Switch to CPU to avoid CUDA/CuDNN issues
        model = WhisperModel("small", device="cpu", compute_type="float32")  # Adjust parameters as needed
        model.sample_rate = RATE  # Set the sample rate attribute for buffer calculations
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.exception(f"Failed to load Whisper model: {e}")
        sys.exit(1)  # Exit the script if model loading fails

    # Open the audio stream
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index,
                        stream_callback=audio_callback)

        logging.info(f"Recording from '{desired_device_name}' with {CHANNELS} channel(s) at {RATE} Hz...")
        stream.start_stream()

        # Start transcription thread
        transcription_thread = threading.Thread(target=transcribe_audio, args=(model, audio_queue, counter))
        transcription_thread.daemon = True
        transcription_thread.start()
        logging.debug("Transcription thread started.")

        logging.info("Listening... Press Ctrl+C to stop.")

        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping...")
    except Exception as e:
        logging.exception(f"An error occurred while opening the stream: {e}")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
            logging.debug("Audio stream stopped and closed.")
        p.terminate()
        logging.debug("PyAudio terminated.")
        sys.exit(0)
