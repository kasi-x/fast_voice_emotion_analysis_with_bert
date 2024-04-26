import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification

# SETTINGS
BLOCKSIZE = 24678 // 5
SILENCE_THRESHOLD = 700
MIN_AUDIO_LENGTH = 8000
SILENCE_RATIO = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Whisper model and processor
model_name = "vumichien/whisper-small-ja"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model = model.half()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

# Initialize BERT model
BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model = bert_model.half()

audio_queue = multiprocessing.Queue()
classification_queue = multiprocessing.Queue()
shared_ndarray_list = None


def audio_capture_thread():
    try:
        print("Starting audio capture thread...")
        with sd.InputStream(
            samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
        ) as stream:
            while True:
                indata, status = stream.read(BLOCKSIZE)
                print(f"Captured audio data: {indata[:10]}")
                audio_queue.put((indata, status))
    except Exception as e:
        print(f"Error in audio_capture_thread: {e}")


def bert_classification():
    try:
        print("Starting BERT classification thread...")
        while True:
            transcription = classification_queue.get()
            inputs = tokenizer(
                transcription,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(device)
            outputs = bert_model(**inputs)
            predicted_label_idx = torch.argmax(outputs.logits, dim=1).item()
            labels = ["very negative", "negative", "neutral", "positive", "very positive"]
            print(f"Predicted emotion: {labels[predicted_label_idx]}")
    except Exception as e:
        print(f"Error in bert_classification: {e}")


def transcription_and_plotting():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.random.randn(BLOCKSIZE))
    ax.set_ylim([-(2**15), 2**15 - 1])
    ax.set_xlim(0, BLOCKSIZE)

    global shared_ndarray_list
    while True:
        indata, status = audio_queue.get()
        indata_flattened = abs(indata.flatten())
        line.set_ydata(indata)
        plt.draw()

        is_significant_audio = (
            np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size >= SILENCE_RATIO
        )
        if is_significant_audio:
            shared_ndarray_list.extend(indata.flatten())
        elif len(shared_ndarray_list) > 0:
            global_ndarray = np.array(shared_ndarray_list, dtype="int16")
            if len(global_ndarray) < MIN_AUDIO_LENGTH:
                continue
            indata_transformed = global_ndarray.astype(np.float32) / 32768.0
            shared_ndarray_list.clear()
            input_data = processor(
                indata_transformed, sampling_rate=16000, return_tensors="pt"
            ).input_features
            input_data = input_data.half()
            predicted_ids = model.generate(
                input_data.to(device), forced_decoder_ids=forced_decoder_ids
            )
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            print(f"Transcription: {transcription}")
            classification_queue.put(transcription[0])


def main():
    global shared_ndarray_list
    manager = multiprocessing.Manager()
    shared_ndarray_list = manager.list()

    capture_process = multiprocessing.Process(target=audio_capture_thread)
    classification_process = multiprocessing.Process(target=bert_classification)

    capture_process.start()
    classification_process.start()

    try:
        transcription_and_plotting()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        capture_process.terminate()
        classification_process.terminate()


if __name__ == "__main__":
    main()
