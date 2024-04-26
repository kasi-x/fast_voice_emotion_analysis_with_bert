import sounddevice as sd
import numpy as np

import matplotlib.pyplot as plt
import whisper
import threading
import asyncio
import queue
import sys
import numpy as np
import sounddevice as sd
import asyncio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import matplotlib.pyplot as plt


# SETTINGS
# MODEL_TYPE = "base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
# LANGUAGE = "English"
# pre-set the language to avoid autodetection
BLOCKSIZE = 24678 // 5
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD = 700
MIN_AUDIO_LENGTH = 8000
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO = 300
# number of samples in one buffer that are allowed to be higher than threshold

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained("clu-ling/whisper-large-v2-japanese-5k-steps")
model = WhisperForConditionalGeneration.from_pretrained(
    "clu-ling/whisper-large-v2-japanese-5k-steps"
).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

global_ndarray = None
# model = whisper.load_model(MODEL_TYPE)


plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot(np.random.randn(BLOCKSIZE))
ax.set_ylim([-(2**15), 2**15 - 1])
ax.set_xlim(0, BLOCKSIZE)

audio_queue = queue.Queue()  # Use a regular Python queue

def audio_capture_thread():
    """Thread that captures audio and puts blocks of data into the audio queue."""
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE) as stream:
        while True:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))

async def inputstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    while True:
        indata, status = audio_queue.get()
        yield indata, status

async def inputstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE, callback=callback
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def process_audio_buffer():
    global global_ndarray
    receiving_audio = False

    async for indata, status in inputstream_generator():
        indata_flattened = abs(indata.flatten())
        line.set_ydata(indata)
        plt.draw()
        plt.pause(0.001)

        # Check if current chunk has significant audio
        is_significant_audio = (
            np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size >= SILENCE_RATIO
        )

        # If it has significant audio
        if is_significant_audio:
            print("Status: Receiving audio data.")
            receiving_audio = True
            if global_ndarray is not None:
                global_ndarray = np.concatenate((global_ndarray, indata), dtype="int16")
            else:
                global_ndarray = indata
            continue

        # If current chunk is silent and there was audio being received previously
        if not is_significant_audio and receiving_audio:
            print("Status: Detected silence after receiving audio.")
            if len(global_ndarray) < MIN_AUDIO_LENGTH:
                print(
                    f"Status: Audio length {len(global_ndarray)} is insufficient. Awaiting more input."
                )
                continue

            print("Status: Processing audio data...")
            indata_transformed = global_ndarray.flatten().astype(np.float32) / 32768.0
            input_features = processor(
                indata_transformed, sampling_rate=16000, return_tensors="pt"
            ).input_features
            predicted_ids = model.generate(
                input_features.to(device), forced_decoder_ids=forced_decoder_ids
            )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            print(f"Transcription: {transcription}")
            global_ndarray = None
            receiving_audio = False
        else:
            print("Status: Detected silence.")


if __name__ == "__main__":
    thread = threading.Thread(target=audio_capture_thread, daemon=True)
    thread.start()
    try:
        asyncio.run(process_audio_buffer())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
