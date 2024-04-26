from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
import numpy as np
import matplotlib.pyplot as plt


model = AutoModelForSequenceClassification.from_pretrained(
    "patrickramos/bert-base-japanese-v2-wrime-fine-tune"
)
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# pipelineでの感情分析結果
results = nlp("私はとっても幸せ")

# readerに関する結果のみをフィルタリング
reader_results = [result for result in results if "reader_" in result["label"]]
values = [result["score"] for result in reader_results]

# 感情の日本語訳
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

# ラベルを日本語に変換
labels = [emotion_translation[result["label"].split("_")[1]] for result in reader_results]

# N-gramの設定 (ここでは2-gram)
N = 2
ngram_values = [np.mean(values[i : i + N]) for i in range(len(values) - N + 1)]
ngram_labels = [f"{labels[i]}-{labels[i+1]}" for i in range(len(labels) - N + 1)]

# プロット
plt.figure(figsize=(10, 7))
plt.bar(range(len(ngram_values)), ngram_values, color="skyblue", align="center")
plt.xticks(range(len(ngram_values)), ngram_labels, rotation=45)
plt.xlabel("N-gramの感情ペア")
plt.ylabel("平均スコア")
plt.title("Readerの感情分析 (N-gram)")
plt.tight_layout()
plt.show()
