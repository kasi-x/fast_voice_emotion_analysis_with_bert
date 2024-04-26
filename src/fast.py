import sounddevice as sd
import numpy as np

import matplotlib.pyplot as plt
import whisper

import asyncio
import queue
import sys


# SETTINGS
MODEL_TYPE = "base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE = "English"
# pre-set the language to avoid autodetection
BLOCKSIZE = 24678 // 5
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD = 700
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO = 2000
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
model = whisper.load_model(MODEL_TYPE)


async def inputstream_generator():
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        print("Received audio data.")  # Log when audio data is received.
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE, callback=callback
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            print(
                f"Yielding {len(indata)} frames of audio data."
            )  # Log the amount of audio data being yielded.
            yield indata, status


plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot(np.random.randn(BLOCKSIZE))
ax.set_ylim([-(2**15), 2**15 - 1])
ax.set_xlim(0, BLOCKSIZE)


async def process_audio_buffer():
    global global_ndarray
    async for indata, status in inputstream_generator():
        indata_flattened = abs(indata.flatten())

        line.set_ydata(indata)
        plt.draw()
        plt.pause(0.001)

        # Log the size of non-silent data.
        non_silent_size = np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size
        print(f"Non-silent data size: {non_silent_size}")

        if non_silent_size < SILENCE_RATIO:
            print("Discarding buffer due to silence.")
            continue

        if global_ndarray is not None:
            global_ndarray = np.concatenate((global_ndarray, indata), dtype="int16")
        else:
            global_ndarray = indata

        avg_end_signal = np.average((indata_flattened[-100:-1]))
        if avg_end_signal > SILENCE_THRESHOLD / 15:
            print("Appending buffer as the end is not silent.")
            continue
        else:
            local_ndarray = global_ndarray.copy()
            global_ndarray = None
            indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
            result = model.transcribe(indata_transformed, language=LANGUAGE)
            print(f"Transcription Result: {result['text']}")  # Log the transcription result.

        del local_ndarray
        del indata_flattened


async def main():
    print("\nActivating wire ...\n")
    audio_task = asyncio.create_task(process_audio_buffer())
    while True:
        await asyncio.sleep(1)
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print("\nwire was cancelled")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
