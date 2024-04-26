import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
import queue

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

# Initialize BERT model and tokenizer for sentiment analysis
BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model = bert_model.half()

global_ndarray = None
audio_queue = queue.Queue()
classification_queue = queue.Queue()


def audio_capture_thread():
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
    ) as stream:
        while True:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))


def bert_classification():
    while True:
        transcription = classification_queue.get()
        inputs = tokenizer(
            transcription, return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        outputs = bert_model(**inputs)
        predicted_label_idx = torch.argmax(outputs.logits, dim=1).item()

        labels = ["very negative", "negative", "neutral", "positive", "very positive"]
        print(f"Predicted emotion: {labels[predicted_label_idx]}")


def transcription_and_plotting():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.random.randn(BLOCKSIZE))
    ax.set_ylim([-(2**15), 2**15 - 1])
    ax.set_xlim(0, BLOCKSIZE)

    global global_ndarray

    while True:
        indata, status = audio_queue.get()
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

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            print(f"Transcription: {transcription}")

            # BERT分類スレッドに転送
            classification_queue.put(transcription[0])


if __name__ == "__main__":
    capture_thread = threading.Thread(target=audio_capture_thread)
    classification_thread = threading.Thread(target=bert_classification)

    capture_thread.start()
    classification_thread.start()

    try:
        transcription_and_plotting()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
