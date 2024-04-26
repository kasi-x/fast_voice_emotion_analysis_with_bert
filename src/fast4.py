import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import queue

# SETTINGS
BLOCKSIZE = 24678 // 5
SILENCE_THRESHOLD = 700
MIN_AUDIO_LENGTH = 8000
SILENCE_RATIO = 300

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# "clu-ling/whisper-large-v2-japanese-5k-steps"
model_name = "vumichien/whisper-small-ja"
# model_name = "kimbochen/whisper-tiny-ja"
# C:\Users\anosillus\.cache\huggingface\hub\models--kimbochen--whisper-tiny-ja
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model = model.half()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

global_ndarray = None
audio_queue = queue.Queue()


def audio_capture_thread():
    """Thread that captures audio and puts blocks of data into the audio queue."""
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
    ) as stream:
        while True:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))


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


if __name__ == "__main__":
    capture_thread = threading.Thread(target=audio_capture_thread)
    capture_thread.start()

    try:
        transcription_and_plotting()
    except KeyboardInterrupt:
        print("\nInterrupted by user")

"""

# ... [上記のコードとインポート文はここに続く]

# 1. モデルとトークナイザの初期化
emotion_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

def predict_emotion(text):
    # 2. `predict_emotion`関数を定義
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
    
    probabilities = softmax(outputs.logits, dim=1)
    class_id = torch.argmax(probabilities).item()
    
    return "Positive" if class_id == 1 else "Negative"

def transcription_and_plotting():
    # ... [関数の中身の初めの部分]

    while True:
        # ... [関数の中のループの初めの部分]

        if is_significant_audio:
            # ... [この部分の残り]

            # 3. トランスクリプトが得られた後、そのトランスクリプトを感情分析関数に渡す
            if transcription:
                emotion_result = predict_emotion(transcription[0])

                # 4. トランスクリプトと感情の結果を表示
                print(f"Transcription: {transcription}")
                print(f"Emotion: {emotion_result}")

# ... [関数の定義の後の部分]
"""
