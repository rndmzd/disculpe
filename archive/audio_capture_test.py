import sounddevice as sd
import numpy as np
import wave
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def list_audio_devices():
    print(sd.query_devices())
    logging.debug("Listed all available audio devices.")


def record_audio(device_name, duration=5, output_filename="test_recording.wav"):
    try:
        device_info = sd.query_devices(device_name, "input")
        channels = 1  # Mono
        samplerate = int(device_info["default_samplerate"])

        logging.debug(
            f"Recording parameters set: Channels={channels}, Samplerate={samplerate}, Duration={duration}s"
        )

        print(f"Recording... Speak into the device '{device_name}'.")
        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            device=device_name,
        )
        sd.wait()  # Wait until recording is finished

        # Save as WAV file
        wf = wave.open(output_filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        wf.close()

        logging.info(f"Audio saved to '{output_filename}'.")
        print(f"Audio saved to '{output_filename}'.")
    except Exception as e:
        logging.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    list_audio_devices()
    device_name = input(
        "Enter the exact name of the device you want to record from (as listed above): "
    ).strip()
    record_duration = 5  # seconds
    output_file = "test_recording.wav"
    record_audio(device_name, duration=record_duration, output_filename=output_file)
