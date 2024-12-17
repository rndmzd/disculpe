import pyaudio
import queue
import threading
from faster_whisper import WhisperModel  # Ensure faster-whisper is installed and compatible
# Alternatively, use another transcription library if faster-whisper is not working
import sys

def list_audio_devices():
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            yield i, device_info['name']
    finally:
        p.terminate()

def get_device_index_by_name(target_name):
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(device_info)
            if device_info['name'] == target_name:
                return i
        return None
    finally:
        p.terminate()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def transcribe_audio(model, audio_queue, counter):
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            # Convert byte data to appropriate format for transcription
            # This step depends on the transcription library's requirements
            # For example, faster-whisper might require a NumPy array
            # Placeholder for actual transcription code:
            transcription = model.transcribe(data)  # Adjust as needed
            if "disculpe" in transcription.lower():
                counter.increment()

class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1
            print(f"Disculpe Count: {self.count}")
            # Update the counter overlay (e.g., write to a file or update a browser source)

if __name__ == "__main__":
    # Initialize PyAudio and select device
    desired_device_name = "Speakers (Steam Streaming Speak"
    device_index = get_device_index_by_name(desired_device_name)
    if device_index is None:
        print(f"Device '{desired_device_name}' not found.")
        sys.exit(1)

    # Initialize transcription model
    model = WhisperModel("small", device="cuda", compute_type="float16")

    # Initialize counter
    counter = Counter()

    # Initialize audio queue
    audio_queue = queue.Queue()

    # Initialize PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024,
                    input_device_index=device_index,
                    stream_callback=audio_callback)

    # Start the stream
    stream.start_stream()

    # Start transcription thread
    transcription_thread = threading.Thread(target=transcribe_audio, args=(model, audio_queue, counter))
    transcription_thread.daemon = True
    transcription_thread.start()

    print("Listening... Press Ctrl+C to stop.")

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Stopping...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(0)
