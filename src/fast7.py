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

audio_queue = queue.Queue()
classification_queue = queue.Queue()
running = True


def initialize_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Whisper model and processor
    model_name = "vumichien/whisper-small-ja"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model = model.half()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

    # BERT model and tokenizer
    BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model = bert_model.half()

    return model, processor, forced_decoder_ids, bert_model, tokenizer, device


def audio_capture_thread():
    global running
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
    ) as stream:
        while running:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))
    audio_queue.put(None)  # Signal termination


def bert_classification(tokenizer, bert_model, device):
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    while True:
        transcription = classification_queue.get()
        if transcription is None:  # Check for termination signal
            break
        inputs = tokenizer(
            transcription, return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        outputs = bert_model(**inputs)
        predicted_label_idx = torch.argmax(outputs.logits, dim=1).item()

        print(f"Predicted emotion: {labels[predicted_label_idx]}")


def process_audio_data(line, global_ndarray, model, processor, forced_decoder_ids, device):
    indata, status = audio_queue.get()
    if indata is None:  # Check for termination signal
        return None, None

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
            return global_ndarray, None
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

        # Send to BERT classification thread
        classification_queue.put(transcription)
    return global_ndarray, None


def update_plot(line):
    global running
    while running:
        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    model, processor, forced_decoder_ids, bert_model, tokenizer, device = initialize_models()

    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.random.randn(BLOCKSIZE))
    ax.set_ylim([-(2**15), 2**15 - 1])
    ax.set_xlim(0, BLOCKSIZE)

    capture_thread = threading.Thread(target=audio_capture_thread)
    classification_thread = threading.Thread(
        target=bert_classification, args=(tokenizer, bert_model, device)
    )
    plot_thread = threading.Thread(target=update_plot, args=(line,))

    capture_thread.start()
    classification_thread.start()
    plot_thread.start()

    global_ndarray = None

    try:
        while running:
            global_ndarray, _ = process_audio_data(
                line, global_ndarray, model, processor, forced_decoder_ids, device
            )
            if global_ndarray is None:
                running = False
    except KeyboardInterrupt:
        running = False
        capture_thread.join()
        classification_thread.join()
        plot_thread.join()  # Make sure to join the plot thread as well
        classification_queue.put(None)  # Signal termination to the classification thread
        print("\nInterrupted by user")
