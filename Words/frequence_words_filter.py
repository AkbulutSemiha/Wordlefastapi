import pandas as pd

df = pd.read_csv("five_letter_word_frequencies.csv")

alpha = 0.5  # frequency öncelikli
df["score"] = (df["frequency"] / df["frequency"].max()) * alpha + df["dispersion"] * (1 - alpha)
answer_pool = df.sort_values("score", ascending=False)
tr_map = str.maketrans("çğiöşü", "ÇĞİÖŞÜ")

answer_pool["word"] = answer_pool["word"].apply(lambda x: x.translate(tr_map).upper())


try:
    with open("words_tr.txt", "r", encoding="utf-8") as file:
        words = [line.strip().upper() for line in file if line.strip()]
except FileNotFoundError:
    words = []
finally:
    file.close()

filtered = answer_pool[answer_pool["word"].isin(words)]
filtered = filtered.head(1000)
