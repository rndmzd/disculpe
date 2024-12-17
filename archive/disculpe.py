import threading
import queue
import sounddevice as sd
from faster_whisper import WhisperModel

# Initialize the model
model = WhisperModel("small", device="cuda", compute_type="float16")

# Audio parameters
sample_rate = 16000
channels = 1
chunk_duration = 5  # seconds
buffer = queue.Queue()


def audio_callback(indata, frames, time, status):
    buffer.put(indata.copy())


def list_audio_devices():
    print("Available Audio Devices:\n")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(device)
        #print(
        #    f"Device {idx}: {device['name']} - {'Input' if device['max_input_channels'] > 0 else 'Output'}"
        #)

list_audio_devices()

device_name = input("\nEnter the name of the virtual audio device: ")
print(f"\nUsing device: {device_name}")

# Start audio stream from virtual device
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    callback=audio_callback,
    device=device_name,
)
stream.start()

counter = 0
counter_file = "counter.txt"


def increment_counter(occurrences):
    global counter
    counter += occurrences
    with open(counter_file, "w", encoding="utf-8") as f:
        f.write(str(counter))


def transcribe_loop():
    global counter
    while True:
        # Collect audio for the chunk duration
        audio_chunks = []
        for _ in range(int(sample_rate * chunk_duration / 1024)):
            audio = buffer.get()
            audio_chunks.append(audio)

        # Concatenate audio chunks
        audio_data = (
            np.concatenate(audio_chunks).flatten().astype(np.float32) / 32768.0
        )  # Normalize if necessary

        # Transcribe using Faster-Whisper
        segments, _ = model.transcribe(audio_data, language="es")
        text = " ".join([segment.text for segment in segments]).lower()

        # Count occurrences of "disculpe"
        occurrences = text.count("disculpe")
        if occurrences > 0:
            increment_counter(occurrences)


# Start transcription in a separate thread
transcription_thread = threading.Thread(target=transcribe_loop)
transcription_thread.start()
