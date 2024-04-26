import asyncio
import numpy as np
import sounddevice as sd
import sys
import threading
import curses
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

import whisper

import asyncio
import queue
import sys


# SETTINGS
MODEL_TYPE = "base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE = "English"
# pre-set the language to avoid autodetection
BLOCKSIZE = 24678
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds.
SILENCE_THRESHOLD = 700
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO = 2000
# number of samples in one buffer that are allowed to be higher than threshold


global_ndarray = None
model = whisper.load_model(MODEL_TYPE)

data_queue = asyncio.Queue()


# Curses UI function
def display_ui(data_queue):
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(1)  # non-blocking input
    try:
        while True:
            stdscr.clear()
            try:
                data = data_queue.get_nowait()
            except:
                data = None

            if data:
                stdscr.addstr(0, 0, data)
            stdscr.refresh()
            curses.napms(100)  # Wait for 100ms
    except KeyboardInterrupt:
        pass
    finally:
        curses.endwin()


# Real-time plotting function
def realtime_plot():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(BLOCKSIZE)
    ax.set_ylim([-(2**15), 2**15 - 1])
    ax.set_xlim(0, BLOCKSIZE)

    while True:
        try:
            indata = plot_queue.get()
            line.set_ydata(indata)
            plt.draw()
            plt.pause(0.001)
        except KeyboardInterrupt:
            break


# Your original inputstream generator here...

# Modified process_audio_buffer function
plot_queue = asyncio.Queue()


async def process_audio_buffer():
    global global_ndarray
    async for indata, status in inputstream_generator():
        indata_flattened = abs(indata.flatten())
        non_silent_data_size = np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size

        message = f"Non-silent data size: {non_silent_data_size} | "

        # Append indata to the plot_queue for real-time plotting
        await plot_queue.put(indata)

        if non_silent_data_size < SILENCE_RATIO:
            message += "Determined as silence. Skipping buffer."
        else:
            message += "Determined as non-silent."
            # ... (Rest of your code)

        await data_queue.put(message)


def main():
    ui_thread = threading.Thread(target=display_ui, args=(data_queue,))
    plot_thread = threading.Thread(target=realtime_plot)
    ui_thread.start()
    plot_thread.start()

    try:
        asyncio.run(process_audio_buffer())
    except KeyboardInterrupt:
        ui_thread.join()
        plot_thread.join()


if __name__ == "__main__":
    main()
