import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from matplotlib.animation import FuncAnimation
import os

plt.rcParams["font.family"] = "Meiryo"

SAVE_PATH = "transcriptions.txt"

tokenizer = AutoTokenizer.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)

emotions = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼", "喜び"]


def get_emotion_probs(text):
    token = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    )
    output = model(**token)
    normalized_logits = (output.logits - torch.min(output.logits)) / (
        torch.max(output.logits) - torch.min(output.logits)
    )
    probs = normalized_logits.squeeze().tolist()
    probs.append(probs[0])
    return probs


fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.set_ylim(0, 1)
theta = np.linspace(0, 2 * np.pi, len(emotions), endpoint=True)
(line,) = ax.plot(theta, [0] * len(emotions))
ax.set_xticks(theta)
ax.set_xticklabels(emotions)

last_read_line = 0
last_mtime = os.path.getmtime(SAVE_PATH)  # 最後に確認したファイルの修正時間

serialPort = "/dev/ttyUSB0"
baudRate = 115200
ser = serial.Serial(serialPort, baudRate, timeout=1)
time.sleep(2)

# 感情と色の対応
emotion_colors = {
    "喜び": "Y",  # 黄色
    "悲しみ": "B",  # 青色
    "期待": "G",  # 緑色
    # 他の感情に対応する色も同様に定義
}


def send_color_to_esp32(color_code):
    ser.write(color_code.encode())


def update(frame):
    global last_read_line, last_mtime

    current_mtime = os.path.getmtime(SAVE_PATH)

    if current_mtime != last_mtime:
        with open(SAVE_PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if last_read_line < len(lines):
                text = lines[-1].strip()
                emotion_probs = get_emotion_probs(text)
                line.set_ydata(emotion_probs)
                last_read_line = len(lines)
                last_mtime = current_mtime

                highest_emotion = emotions[emotion_probs.index(max(emotion_probs))]
                color_code = emotion_colors.get(highest_emotion, "R")
                send_color_to_esp32(color_code)  # ESP32に色を送信

    return (line,)


MAX_FRAMES = 100

ani = FuncAnimation(fig, update, repeat=True, blit=True, save_count=MAX_FRAMES)

plt.show()
