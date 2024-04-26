import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

plt.rcParams["font.family"] = "Meiryo"

# モデルとトークナイザの準備
tokenizer = AutoTokenizer.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)

# 感情のリスト
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


fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="polar")
ax.set_ylim(0, 1)

theta = np.linspace(0, 2 * np.pi, len(emotions) + 1, endpoint=True)  # 最後に最初の値を追加
(l,) = ax.plot([], [])

texts = ["すごく楽しかった。"]
data = get_emotion_probs(texts[0])


