import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import threading
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
import queue
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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

BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model = bert_model.half()

global_ndarray = None
audio_queue = queue.Queue()
classification_queue = queue.Queue()

# 追加: スレッドの動作を制御するフラグ
running = True


sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "patrickramos/bert-base-japanese-v2-wrime-fine-tune"
).to(device)
sentiment_model = sentiment_model.half()
sentiment_tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

emotion_translation = {
    "surprise": "驚き",
    "sadness": "悲しみ",
    "fear": "恐れ",
    "disgust": "嫌悪",
    "anger": "怒り",
    "anticipation": "期待",
    "joy": "喜び",
    "trust": "信頼",
}


def audio_capture_thread():
    with sd.InputStream(
        samplerate=16000, channels=1, dtype="int16", blocksize=BLOCKSIZE
    ) as stream:
        while running:
            indata, status = stream.read(BLOCKSIZE)
            audio_queue.put((indata, status))

    audio_queue.put(("STOP", None))


plotting_queue = queue.Queue()


def bert_classification():
    while True:
        transcription = classification_queue.get()
        if transcription == "STOP":
            break

        # Sentiment analysis with the provided model
        results = sentiment_tokenizer(
            transcription, return_tensors="pt", truncation=True, padding=True, max_length=256
        ).to(device)
        outputs = sentiment_model(**results)
        sentiment_results = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()

        reader_results = [
            {"label": label.item(), "score": score}
            for label, score in zip(
                outputs.logits.argmax(dim=1).cpu().numpy(), sentiment_results[0]
            )
            if "reader_" in str(label.item())
        ]
        values = [result["score"] for result in reader_results]

        # ラベルを日本語に変換
        labels = [emotion_translation[result["label"]] for result in reader_results]

        # N-gramの設定 (ここでは2-gram)
        N = 2
        ngram_values = [np.mean(values[i : i + N]) for i in range(len(values) - N + 1)]
        ngram_labels = [f"{labels[i]}-{labels[i+1]}" for i in range(len(labels) - N + 1)]

        # データをキューに追加
        plotting_queue.put((ngram_values, ngram_labels))



emotions = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]

def get_emotion_probs(text):
    token = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    )
    output = model(**token)
    normalized_logits = (output.logits - torch.min(output.logits)) / (
        torch.max(output.logits) - torch.min(output.logits)
    )
    probs = normalized_logits.squeeze().tolist()
    probs.append(probs[0])  # 最初の確率を最後にも追加
    return probs

def main_plotting():
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylim([-(2**15), 2**15 - 1])
    ax1.set_xlim(0, BLOCKSIZE)
    (line,) = ax1.plot(np.zeros(BLOCKSIZE), 'g-')

    ax2 = fig.add_subplot(2, 1, 2, polar=True)
    ax2.set_ylim(0, 1)
    theta = np.linspace(0, 2 * np.pi, len(emotions) + 1, endpoint=True)
    (l,) = ax2.plot([], [])

    index = 0

    def update(i):
        global index
        # 音声データの取得と更新
        indata, _ = audio_queue.get()
        if isinstance(indata, str) and indata == "STOP":
            return
        
        line.set_ydata(indata)
        
        # 以下の部分は、感情分析のためのテキストデータを取得するものと仮定しています。
        # もし実際にテキストデータがキューに入れられる場合は、以下の行を有効にしてください。
        text = classification_queue.get()

        # この例では、固定のテキストリストからデータを取得します。
        text = texts[index % len(texts)]
        index += 1
        
        data = get_emotion_probs(text)

        ax2.clear()
        ax2.set_xticks(theta)
        ax2.set_xticklabels(emotions + [emotions[0]])  # ラベルも最初のものを最後に追加
        ax2.set_ylim(0, 1)
        (l,) = ax2.plot(theta, data, "r-", lw=2)

    ani = FuncAnimation(fig, update, interval=1000, blit=False)
    plt.show()


# def main_plotting():
#     plt.ion()
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
# 
#     ax1.set_ylim([-(2**15), 2**15 - 1])
#     ax1.set_xlim(0, BLOCKSIZE)
#     (line,) = ax1.plot(np.zeros(BLOCKSIZE), 'g-')
# 
#     # レーダーチャートの初期設定
#     emotions = list(emotion_translation.values())
#     num_vars = len(emotions)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     ax2.set_theta_offset(np.pi / 2)
#     ax2.set_theta_direction(-1)
#     ax2.set_rlabel_position(115)
#     ax2.set_xticks(angles)
#     ax2.set_xticklabels(emotions)
#     ax2.set_ylim(0, 1)
# 
#     while running:
#         # 音声データの取得と更新
#         indata, _ = audio_queue.get()
#         if isinstance(indata, str) and indata == "STOP":
#             break
#         line.set_ydata(indata)
# 
#         # 感情分析データの取得とレーダーチャートの更新
#         ngram_values, ngram_labels = plotting_queue.get()
#         ax2.clear()
#         ax2.set_xticks(angles)
#         ax2.set_xticklabels(emotions)
#         ax2.set_ylim(0, 1)
#         ax2.plot(angles, ngram_values, color='b', linewidth=2, linestyle='solid')
#         ax2.fill(angles, ngram_values, color='skyblue', alpha=0.4)
# 
#         plt.pause(0.001)
# 
#     plt.close()


if __name__ == "__main__":
    capture_thread = threading.Thread(target=audio_capture_thread)
    classification_thread = threading.Thread(target=bert_classification)

    capture_thread.start()
    classification_thread.start()

    try:
        print('start')
        main_plotting()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        running = False

    capture_thread.join()
    classification_thread.join()
