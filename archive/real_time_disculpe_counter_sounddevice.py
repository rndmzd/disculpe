import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import time
import logging
from faster_whisper import WhisperModel

# ---------------------------- Logging Configuration ----------------------------

# Configure logging to output DEBUG and higher level messages to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler(
            "real_time_disculpe_counter_sounddevice.log", mode="w"
        ),  # Log to file
    ],
)

# ---------------------------- Helper Classes ----------------------------


class Counter:
    """
    Thread-safe counter to track the number of times 'disculpe' is detected.
    """

    def __init__(self, counter_file="counter.txt"):
        self.count = 0
        self.lock = threading.Lock()
        self.counter_file = counter_file
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
            logging.debug(
                f"Counter file '{self.counter_file}' updated with count: {self.count}"
            )
        except Exception as e:
            logging.error(f"Failed to update counter file: {e}")


# ---------------------------- Audio Processing Functions ----------------------------


def list_audio_devices():
    """
    Lists all available audio devices with detailed information.
    """
    try:
        devices = sd.query_devices()
        print("Available Audio Devices:\n")
        for idx, device in enumerate(devices):
            host_api = sd.query_hostapis(device["hostapi"])["name"]
            print(f"Device {idx}: {device['name']}")
            print(f"    Host API: {host_api}")
            print(f"    Max Input Channels: {device['max_input_channels']}")
            print(f"    Max Output Channels: {device['max_output_channels']}")
            print(f"    Default Sample Rate: {int(device['default_samplerate'])} Hz\n")

            # Log device details
            logging.debug(
                f"Device {idx}: {device['name']}, Host API: {host_api}, "
                f"Max Input Channels: {device['max_input_channels']}, "
                f"Max Output Channels: {device['max_output_channels']}, "
                f"Default Sample Rate: {int(device['default_samplerate'])} Hz"
            )
    except Exception as e:
        logging.exception(f"Failed to list audio devices: {e}")


def get_device_index(device_name):
    """
    Retrieves the device index for a given device name.
    If multiple devices have the same name, lists them with their indices and host APIs.
    Returns None if the device is not found.
    """
    try:
        devices = sd.query_devices()
        matching_devices = []
        for idx, device in enumerate(devices):
            if device["name"] == device_name:
                host_api = sd.query_hostapis(device["hostapi"])["name"]
                matching_devices.append((idx, host_api))

        if not matching_devices:
            logging.error(f"No devices found with the name '{device_name}'.")
            return None
        elif len(matching_devices) == 1:
            device_index = matching_devices[0][0]
            logging.debug(
                f"Selected device '{device_name}' with index {device_index} and Host API {matching_devices[0][1]}."
            )
            return device_index
        else:
            print(f"Multiple devices found with the name '{device_name}':")
            for idx, host_api in matching_devices:
                print(f"[{idx}] Host API: {host_api}")
            selected_index = input("Enter the device index you want to use: ").strip()
            if not selected_index.isdigit():
                logging.error("Invalid input. Please enter a numeric device index.")
                return None
            selected_index = int(selected_index)
            if any(idx == selected_index for idx, _ in matching_devices):
                logging.debug(
                    f"Selected device '{device_name}' with index {selected_index}."
                )
                return selected_index
            else:
                logging.error(
                    f"Device index {selected_index} is not among the matching devices."
                )
                return None
    except Exception as e:
        logging.exception(f"Error retrieving device index: {e}")
        return None


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
            # logging.debug("Received audio data from queue.")
            # Convert byte data to numpy array (float32 normalized)
            audio_data = data.flatten().astype(np.float32) / np.iinfo(np.int16).max
            buffer = np.concatenate((buffer, audio_data))
            # logging.debug(f"Buffer size: {len(buffer)} samples.")

            # Check if buffer has enough data for target_duration
            if len(buffer) >= model.sample_rate * target_duration:
                # Extract chunk for transcription
                chunk = buffer[: model.sample_rate * target_duration]
                buffer = buffer[model.sample_rate * target_duration :]
                # logging.debug(f"Processing chunk of {len(chunk)} samples.")

                # Transcribe
                segments, info = model.transcribe(chunk, language="en", beam_size=5)
                text = " ".join([segment.text for segment in segments]).lower()
                logging.debug(f"Transcribed Text: '{text}'")

                # Check for keyword
                if "disculpe" in text:
                    occurrences = text.count("disculpe")
                    counter.increment(occurrences)
                    logging.info(
                        f"Detected 'disculpe' {occurrences} time(s). Total: {counter.count}"
                    )
        except Exception as e:
            logging.exception(f"Error in transcription thread: {e}")


# ---------------------------- Main Execution ----------------------------


def main():
    # Step 1: List all audio devices
    list_audio_devices()

    # Step 2: Prompt user to select the device by name
    # device_name = input(
    #     "Enter the exact name of the device you want to record from (as listed above): "
    # ).strip()
    device_name = "CABLE Output (VB-Audio Virtual Cable)"

    # Step 3: Get device index (handles multiple devices with the same name)
    device_index = get_device_index(device_name)

    if device_index is None:
        logging.error("No valid device selected. Exiting the script.")
        print("No valid device selected. Exiting the script.")
        sys.exit(1)

    try:
        device_info = sd.query_devices(device_index)
    except Exception as e:
        logging.exception(f"Failed to query device information: {e}")
        print(f"Failed to query device information: {e}")
        sys.exit(1)

    # Step 4: Define recording parameters
    FORMAT = "int16"  # 16-bit resolution
    CHANNELS = 1  # Mono recording
    RATE = int(device_info["default_samplerate"])  # Sample rate
    CHUNK = 1024  # Buffer size
    TARGET_DURATION = 1  # seconds per transcription chunk

    logging.debug(
        f"Recording parameters set: FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}"
    )

    # Step 5: Initialize Whisper model
    # If you encounter CUDA/CuDNN errors, switch to CPU by setting device="cpu" and compute_type="float32"
    try:
        model = WhisperModel(
            "small", device="cpu", compute_type="float32"
            #"small", device="cuda", compute_type="float16"
        )  # Change to "cuda" if GPU is available and properly configured
        model.sample_rate = (
            RATE  # Set the sample_rate attribute for buffer calculations
        )
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.exception(f"Failed to load Whisper model: {e}")
        print(f"Failed to load Whisper model: {e}")
        sys.exit(1)  # Exit the script if model loading fails

    # Step 6: Initialize audio queue and counter
    audio_queue = queue.Queue()
    counter = Counter()

    # Step 7: Start transcription thread
    transcription_thread = threading.Thread(
        target=transcribe_audio, args=(model, audio_queue, counter)
    )
    transcription_thread.daemon = (
        True  # Daemonize thread to exit when main thread exits
    )
    transcription_thread.start()
    logging.debug("Transcription thread started.")

    # Step 8: Define the callback function for sounddevice InputStream
    def audio_callback(indata, frames, time_info, status):
        if status:
            logging.warning(f"Stream status: {status}")
        audio_queue.put(indata.copy())

    # Step 9: Open the audio stream
    try:
        with sd.InputStream(
            samplerate=RATE,
            blocksize=CHUNK,
            dtype=FORMAT,
            channels=CHANNELS,
            callback=audio_callback,
            device=device_index,
        ):
            logging.info(
                f"Recording from '{device_name}' with {CHANNELS} channel(s) at {RATE} Hz..."
            )
            print(f"Recording... Press Ctrl+C to stop.")

            while True:
                time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping...")
        print("\nStopping the recording...")
    except Exception as e:
        logging.exception(f"An error occurred while opening the stream: {e}")
        print(f"An error occurred while opening the stream: {e}")
    finally:
        logging.debug("Audio stream closed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
