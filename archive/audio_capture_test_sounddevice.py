import sounddevice as sd
import numpy as np
import wave
import logging
import sys

# ---------------------------- Logging Configuration ----------------------------

# Configure logging to output DEBUG and higher level messages to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler(
            "audio_capture_test_sounddevice.log", mode="w"
        ),  # Log to file
    ],
)

# ---------------------------- Helper Functions ----------------------------


def list_audio_devices():
    """
    Lists all available audio devices with detailed information.
    """
    try:
        devices = sd.query_devices()
        print("Available Audio Devices:\n")
        for idx, device in enumerate(devices):
            print(f"Device {idx}: {device['name']}")
            print(f"    Host API: {sd.query_hostapis(device['hostapi'])['name']}")
            print(f"    Max Input Channels: {device['max_input_channels']}")
            print(f"    Max Output Channels: {device['max_output_channels']}")
            print(f"    Default Sample Rate: {int(device['default_samplerate'])} Hz\n")

            # Log device details
            logging.debug(
                f"Device {idx}: {device['name']}, Host API: {sd.query_hostapis(device['hostapi'])['name']}, "
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


# ---------------------------- Audio Recording Function ----------------------------


def record_audio(device_index, duration=5, output_filename="test_recording.wav"):
    """
    Records audio from the specified device index for a given duration and saves it to a WAV file.

    :param device_index: Index of the audio device to record from.
    :param duration: Duration in seconds to record.
    :param output_filename: Name of the output WAV file.
    """
    try:
        device_info = sd.query_devices(device_index)
        if device_info["max_input_channels"] < 1:
            logging.error(
                f"Device index {device_index} ('{device_info['name']}') does not support input channels."
            )
            print(
                f"Error: Device '{device_info['name']}' does not support input channels."
            )
            return

        CHANNELS = 1  # Mono recording
        RATE = int(device_info["default_samplerate"])  # Sample rate
        CHUNK = 1024  # Buffer size

        logging.debug(
            f"Recording parameters set: Channels={CHANNELS}, Samplerate={RATE}, Duration={duration}s"
        )

        print(f"Recording... Speak into the device '{device_info['name']}'.")
        logging.info(
            f"Recording started from device index {device_index} ('{device_info['name']}') for {duration} seconds."
        )

        recording = sd.rec(
            int(duration * RATE),
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            device=device_index,
        )
        sd.wait()  # Wait until recording is finished

        # Save as WAV file
        wf = wave.open(output_filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(RATE)
        wf.writeframes(recording.tobytes())
        wf.close()

        logging.info(f"Audio saved to '{output_filename}'.")
        print(f"Audio saved to '{output_filename}'.")

    except KeyboardInterrupt:
        logging.info("Recording interrupted by user.")
        print("\nRecording interrupted by user.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during recording: {e}")


# ---------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    # Step 1: List all audio devices
    list_audio_devices()

    # Step 2: Prompt user to select the device by name
    device_name = input(
        "Enter the exact name of the device you want to record from (as listed above): "
    ).strip()

    # Step 3: Get device index (handles multiple devices with the same name)
    device_index = get_device_index(device_name)

    if device_index is not None:
        # Step 4: Define recording parameters
        recording_duration = 5  # seconds
        output_file = "test_recording.wav"

        # Step 5: Start recording
        record_audio(
            device_index, duration=recording_duration, output_filename=output_file
        )
    else:
        logging.error("No valid device selected. Exiting the test script.")
        print("No valid device selected. Exiting the test script.")
