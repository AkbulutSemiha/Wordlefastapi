
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CSV verisini okuma
data = pd.read_csv("../merged_result.csv")

# Yöntem isimleri
methods = ["Rule-Based Prediction","Hybrid Prediction_Euclidean","Entropy Prediction"]

# Target Word'e göre gruplama ve eşleşmeme sayılarının hesaplanması
group_mismatch_counts = {}

for target, group in data.groupby("Target Word"):
    group_mismatch_counts[target] = {method: (group["Target Word"] != group[method]).sum() for method in methods}

# Sonuçları DataFrame olarak gösterme
group_mismatch_df = pd.DataFrame.from_dict(group_mismatch_counts, orient="index")
group_mismatch_df.index.name = "Target Word"

# 6 ve daha büyük olan değerleri sayma
high_mismatch_counts = (group_mismatch_df >= 6).sum()

# Sonuçları ekrana yazdırma
print(group_mismatch_df)
print("\n6 ve daha büyük eşleşmeme sayıları:")
print(high_mismatch_counts)


# 6 ve daha büyük olan eşleşmeme sayılarının bar chart (çubuk grafik) ile gösterimi
plt.figure(figsize=(6,4))
high_mismatch_counts.plot(kind='bar', color='darkseagreen', edgecolor='black')
plt.title("High Mismatch Counts (>= 6) per Target Word")
plt.xlabel("Target Word")
plt.ylabel("Count of Methods with >= 6 Mismatches")
plt.xticks(rotation=45)
plt.ylim(0, high_mismatch_counts.max() + 10)

plt.tight_layout()
plt.show()
# Her yöntemin ortalama başarıya ulaştığı adım sayısı
def average_steps_per_method(prediction_column):
    # Doğru tahmin edilenler için minimum adım sayısını gruplama
    steps_per_word = data[data['Target Word'] == data[prediction_column]].groupby('Target Word')['Steps'].min()

    # Minimum adım sayılarının ortalamasını al
    return steps_per_word.mean()

ai_avg_steps = average_steps_per_method("AI Prediction")
hybrid_avg_steps = average_steps_per_method("Hybrid Prediction")
rule_based_avg_steps = average_steps_per_method("Rule-Based Prediction")
entropy_avg_steps = average_steps_per_method("Entropy Prediction")

print(f"AI : {ai_avg_steps:.2f}")
print(f"Rule-Based : {rule_based_avg_steps:.2f}")
print(f"Entropy: {entropy_avg_steps:.2f}")
# Yöntemler ve ortalama adım sayıları
methods = ["AI Prediction","Hybrıd Prediction", "Rule-Based Prediction", "Entropy Prediction"]
avg_steps = [ai_avg_steps, hybrid_avg_steps,rule_based_avg_steps, entropy_avg_steps]

# Çubuk grafik
plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=avg_steps, palette='viridis')

# Grafik başlıkları ve etiketleri
plt.title('Average Number of Steps for Successful Prediction', fontsize=16)
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Average Number of Steps', fontsize=12)

# Y eksenini sınırlandırma ve aralık ekleme
plt.ylim(0, 5)  # Y eksenini 0-5 arası sınırla
plt.yticks(range(0, 6, 1))  # 0'dan 5'e kadar tam sayılar

# Çubukların üstüne değer yazdırma
for i, v in enumerate(avg_steps):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=12)

# Göster
plt.show()


# Adım 8: Adım Başına Doğru ve Yanlış Tahmin Oranı
def step_accuracy_rate(method_column):
    first_correct_steps = (
        data[data['Target Word'] == data[method_column]]
        .groupby('Target Word')['Steps']
        .min()
    )

    # Adımlara göre doğru tahmin edilen kelimeleri gruplama
    step_accuracy = (
            first_correct_steps.value_counts(normalize=True) * 100
    )

    # Tüm adımları sıralamak için yeniden indeksleme
    step_accuracy = step_accuracy.sort_index()

    return step_accuracy

hybrid_accuracy = step_accuracy_rate("Hybrid Prediction")
rule_based_accuracy = step_accuracy_rate('Rule-Based Prediction')
entropy_accuracy = step_accuracy_rate('Entropy Prediction')

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.lineplot(data=hybrid_accuracy, label="Hybrid Accuracy", marker='P', markersize=10, linewidth=2, markeredgewidth=2)
sns.lineplot(data=rule_based_accuracy, label="Rule-Based Accuracy", marker='D', markersize=10, linewidth=2,  markeredgewidth=2)
sns.lineplot(data=entropy_accuracy, label="Entropy Accuracy", marker='H', markersize=10, linewidth=2, markeredgewidth=2)

plt.title('Ratio of Correct and Incorrect Predictions per Step')
plt.xlabel('Number of Steps')
plt.ylabel('Success Percentage')
plt.legend()
plt.grid(True)
plt.show()

# Heatmap için başarı oranları
step_accuracy_matrix = pd.DataFrame({
    'AI': ai_accuracy,
    "Hybrid": hybrid_accuracy,
    'Rule-Based': rule_based_accuracy,
    'Entropy': entropy_accuracy
})
plt.figure(figsize=(10, 6))
sns.heatmap(step_accuracy_matrix.T, annot=True, cmap="Blues", fmt=".2f")
plt.title('Heatmap of Correct Prediction Ratio per Step')
plt.xlabel('Number of Steps')
plt.ylabel('Method')
plt.show()
from wordcloud import WordCloud

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["Hybrid Prediction"]])
wordcloud = WordCloud(width=800, height=400,
                      background_color='white'  # Arka planı beyaz yapar
                      ).generate(" ".join(all_predictions))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["Rule-Based Prediction"]])
wordcloud = WordCloud(width=800, height=400,background_color='white').generate(" ".join(all_predictions))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["Entropy Prediction"]])
wordcloud = WordCloud(width=800, height=400,background_color='white').generate(" ".join(all_predictions))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# En iyi ve en kötü performans gösteren kelimeleri listeleme
#
word_steps = data[data['Target Word'] == data['AI Prediction']].groupby('Target Word')['Steps'].min()

# Doğru tahmin edilemeyen kelimeler için NaN değeri atanır
all_words = data['Target Word'].unique()
word_steps = word_steps.reindex(all_words, fill_value=np.nan)

# En iyi performans gösteren kelimeler (ortalama adım sayısına göre)
ai_best_words = word_steps.sort_values().head(10).index.tolist()

# En kötü performans gösteren kelimeler (ortalama adım sayısına göre)
ai_worst_words = word_steps.sort_values(ascending=False).head(10).index.tolist()

word_steps = data[data['Target Word'] == data['Rule-Based Prediction']].groupby('Target Word')['Steps'].min()

# Doğru tahmin edilemeyen kelimeler için NaN değeri atanır
all_words = data['Target Word'].unique()
word_steps = word_steps.reindex(all_words, fill_value=np.nan)

# En iyi performans gösteren kelimeler (ortalama adım sayısına göre)
rb_best_words = word_steps.sort_values().head(10).index.tolist()

# En kötü performans gösteren kelimeler (ortalama adım sayısına göre)
rb_worst_words = word_steps.sort_values(ascending=False).head(10).index.tolist()

word_steps = data[data['Target Word'] == data['Entropy Prediction']].groupby('Target Word')['Steps'].min()

# Doğru tahmin edilemeyen kelimeler için NaN değeri atanır
all_words = data['Target Word'].unique()
word_steps = word_steps.reindex(all_words, fill_value=np.nan)

# En iyi performans gösteren kelimeler (ortalama adım sayısına göre)
en_best_words = word_steps.sort_values().head(10)

# En kötü performans gösteren kelimeler (ortalama adım sayısına göre)
en_worst_words = word_steps.sort_values(ascending=False).head(10)
# Listeleri pandas Series'e dönüştürün
ai_best_words_series = pd.Series(ai_best_words, name='AI Best Words')
rb_best_words_series = pd.Series(rb_best_words, name='Rule-Based Best Words')
en_best_words_series = pd.Series(en_best_words, name='Entropy Best Words')

# Şimdi Series'leri birleştirebilirsiniz
df = pd.concat([ai_best_words_series, rb_best_words_series, en_best_words_series], axis=1)

# Sonuçları gösterin
print(df)


