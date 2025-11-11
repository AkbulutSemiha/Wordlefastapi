import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon
import numpy as np

VOWELS = "aeıioöuüAEIİOÖUÜ"

def cv_pattern(word):
    return ''.join('V' if ch in VOWELS else 'C' for ch in word)

def pattern_distribution(word_list):
    counts = Counter(cv_pattern(w) for w in word_list if w)
    total = sum(counts.values())
    df = pd.DataFrame([{"pattern": p, "count": c, "pct": c/total} for p, c in counts.items()])
    return df.sort_values("pct", ascending=False).reset_index(drop=True)

# === 3. VERİLERİ YÜKLE ===
df_model = pd.read_csv("frequence_words_results.csv")
"""with open("Words/words_tr.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines() if line.strip()]

df_real = pd.DataFrame(words, columns=["words"])"""

model_words = df_model["AI Prediction"].dropna().tolist()
real_words = df_model["Target Word"].dropna().tolist()#df_real["words"].dropna().tolist()

# === 4. DAĞILIMLAR ===
df_model_pat = pattern_distribution(model_words)
df_real_pat = pattern_distribution(real_words)

# === 5. BİRLEŞTİRME ===
df = pd.merge(df_real_pat, df_model_pat, on="pattern", how="outer", suffixes=("_real", "_model")).fillna(0)

# Sayıları tam sayı yap (bazı testler bunu bekler)
df["count_real"] = df["count_real"].astype(int)
df["count_model"] = df["count_model"].astype(int)

# === 6. Normalize ve Test ===
patterns = sorted(set(df['pattern']))
p = np.array(df.set_index('pattern').reindex(patterns).fillna(0)['pct_real'])
q = np.array(df.set_index('pattern').reindex(patterns).fillna(0)['pct_model'])

js_distance = jensenshannon(p, q)
print(f"\n🔹 Jensen–Shannon Distance: {js_distance:.4f} (0=aynı, 1=farklı)")

# Toplamları eşitle (sum farkını önler)
real_counts = np.array(df['count_real'])
model_counts = np.array(df['count_model'])

real_counts = real_counts / real_counts.sum() * len(model_counts)
model_counts = model_counts / model_counts.sum() * len(model_counts)

chi2, pval = chisquare(f_obs=model_counts, f_exp=real_counts)
print(f"🔹 Chi-square test: χ²={chi2:.4f}, p-value={pval:.4f}")

# === 7. GRAFİK ===
top_k = 50
df_top = df.head(top_k)
plt.figure(figsize=(12,5))
plt.bar(df_top["pattern"], df_top["pct_real"], alpha=0.7, label="Gerçek Türkçe")
plt.bar(df_top["pattern"], df_top["pct_model"], alpha=0.7, label="Model")
plt.title("AI Modeli vs Gerçek Türkçe – Fonetik (C/V) Kalıp Dağılımı")
plt.ylabel("Frekans Oranı")
plt.xlabel("C/V Kalıbı")
plt.legend()
plt.tight_layout()
plt.show()

# === 8. SONUÇ TABLOSU ===
print("\nEn yaygın fonetik kalıplar:")
print(df_top[["pattern","pct_real","pct_model"]])
