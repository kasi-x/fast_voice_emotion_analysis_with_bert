import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 感情の日本語訳
emotion_translation = {
    "surprise": "驚き",
    "sadness": "悲しみ",
    "fear": "恐れ",
    "disgust": "嫌悪",
    "anger": "怒り",
    "anticipation": "期待",
    "joy": "喜び",
    "trust": "信頼"
}

# readerの感情のデータ (仮のデータを設定)
labels = list(emotion_translation.values())
values = [0.073, 0.075, 0.076, 0.041, 0.023, 0.022, 0.022, 0.020]
# データの最初の値を末尾に追加して閉じる
values.append(values[0])

# アニメーションの設定
fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.set_ylim(0, 0.1)

theta = np.linspace(0, 2 * np.pi, len(values), endpoint=True)
line, = ax.plot(theta, values, "o-", lw=3)
ax.set_thetagrids(np.arange(0, 360, 360/len(labels)), labels)

def animate(i):
    values_shifted = np.roll(values, shift=i)
    line.set_ydata(values_shifted)
    return line,

ani = FuncAnimation(fig, animate, frames=len(values)-1, repeat=True, blit=True)
plt.show()