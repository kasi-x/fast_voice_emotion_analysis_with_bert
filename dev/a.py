import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import torch
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import queue

# SETTINGS
BLOCKSIZE = 24678 // 5
SILENCE_THRESHOLD = 700
MIN_AUDIO_LENGTH = 8000
SILENCE_RATIO = 300
SAVE_PATH = "transcriptions.txt"

# Initialize Whisper model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vumichien/whisper-small-ja"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model = model.half()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

global_ndarray = None
audio_queue = queue.Queue()

running = True


def audio_capture_thread():
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
    ) as stream:
        while running:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put(indata)

    audio_queue.put(None)  # Sentinel value to indicate end of stream


def transcription_and_plotting():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.random.randn(BLOCKSIZE))
    ax.set_ylim([-(2**15), 2**15 - 1])
    ax.set_xlim(0, BLOCKSIZE)

    global global_ndarray

    while running:
        indata = audio_queue.get()
        if indata is None:  # If end of stream sentinel is found, break the loop
            break

        indata_flattened = abs(indata.flatten())

        line.set_ydata(indata)
        plt.draw()
        plt.pause(0.001)

        is_significant_audio = (
            np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size >= SILENCE_RATIO
        )

        if is_significant_audio:
            if global_ndarray is not None:
                global_ndarray = np.concatenate((global_ndarray, indata), dtype="int16")
            else:
                global_ndarray = indata
        elif global_ndarray is not None:
            if len(global_ndarray) < MIN_AUDIO_LENGTH:
                continue
            indata_transformed = global_ndarray.flatten().astype(np.float32) / 32768.0
            global_ndarray = None
            input_data = processor(
                indata_transformed, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_data = input_data.half()
            predicted_ids = model.generate(
                input_data.to(device), forced_decoder_ids=forced_decoder_ids
            )

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Transcription: {transcription}")

            # with open(SAVE_PATH, "a", encoding="utf-8", buffering=0) as file:
            with open(SAVE_PATH, "a", encoding="utf-8") as file:
                file.write(transcription + "\n")
                file.flush()


if __name__ == "__main__":
    capture_thread = threading.Thread(target=audio_capture_thread)
    capture_thread.start()

    try:
        transcription_and_plotting()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        running = False
        plt.close()

    capture_thread.join()
