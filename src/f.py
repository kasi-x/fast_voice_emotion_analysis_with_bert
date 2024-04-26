
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sounddevice as sd
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import queue

plt.rcParams["font.family"] = "Meiryo"

# SETTINGS
BLOCKSIZE = 24678 // 5
SILENCE_THRESHOLD = 700
MIN_AUDIO_LENGTH = 8000
SILENCE_RATIO = 300

# Initialize Whisper model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vumichien/whisper-small-ja"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model = model.half()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

# Initialize sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime").to(device)

# Lists
emotions = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]

audio_queue = queue.Queue()
global_ndarray = None

running = True

def get_emotion_probs(text):
    token = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    )
    output = sentiment_model(**token)
    normalized_logits = (output.logits - torch.min(output.logits)) / (
        torch.max(output.logits) - torch.min(output.logits)
    )
    probs = normalized_logits.squeeze().tolist()
    probs.append(probs[0])  # 最初の確率を最後にも追加
    return probs

def audio_capture_thread():
    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE) as stream:
        while running:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))

def update_plot(i):
    global global_ndarray
    
    indata, _ = audio_queue.get_nowait()
    
    line.set_ydata(indata)
    
    indata_flattened = abs(indata.flatten())
    is_significant_audio = np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size >= SILENCE_RATIO

    if is_significant_audio:
        if global_ndarray is not None:
            global_ndarray = np.concatenate((global_ndarray, indata), dtype="int16")
        else:
            global_ndarray = indata
    elif global_ndarray is not None:
        if len(global_ndarray) < MIN_AUDIO_LENGTH:
            return
        indata_transformed = global_ndarray.flatten().astype(np.float32) / 32768.0
        global_ndarray = None
        input_data = processor(indata_transformed, sampling_rate=16000, return_tensors="pt").input_features
        input_data = input_data.half()
        predicted_ids = model.generate(input_data.to(device), forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        data = get_emotion_probs(transcription[0])
        radar_line.set_ydata(data)

if __name__ == "__main__":
    fig, axs = plt.subplots(2)
    
    # Audio waveform plot
    (line,) = axs[0].plot(np.random.randn(BLOCKSIZE))
    axs[0].set_ylim([-(2**15), 2**15 - 1])
    axs[0].set_xlim(0, BLOCKSIZE)

    # Sentiment radar chart
    theta = np.linspace(0, 2 * np.pi, len(emotions) + 1, endpoint=True)
    (radar_line,) = axs[1].plot(theta, [0] * (len(emotions) + 1))
    axs[1].set_ylim(0, 1)

    capture_thread = threading.Thread(target=audio_capture_thread)
    capture_thread.start()

    ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False)

    plt.show()

    running = False
    capture_thread.join()