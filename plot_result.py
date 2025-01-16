import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CSV verisini okuma
data = pd.read_csv("16_Ocakwordle_simulations.csv")


# Her yöntemin ortalama başarıya ulaştığı adım sayısı
def average_steps_per_method(prediction_column):
    # Doğru tahmin edilenler için minimum adım sayısını gruplama
    steps_per_word = data[data['Target Word'] == data[prediction_column]].groupby('Target Word')['Steps'].min()

    # Minimum adım sayılarının ortalamasını al
    return steps_per_word.mean()

ai_avg_steps = average_steps_per_method("AI Prediction")
rule_based_avg_steps = average_steps_per_method("Rule-Based Prediction")
entropy_avg_steps = average_steps_per_method("Entropy Prediction")

print(f"AI : {ai_avg_steps:.2f}")
print(f"Rule-Based : {rule_based_avg_steps:.2f}")
print(f"Entropy: {entropy_avg_steps:.2f}")
# Yöntemler ve ortalama adım sayıları
methods = ['AI Method', 'Rule-Based Method', 'Entropy Method']
avg_steps = [ai_avg_steps, rule_based_avg_steps, entropy_avg_steps]

# Çubuk grafik
plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=avg_steps, palette='viridis')

# Grafik başlıkları ve etiketleri
plt.title('Başarılı Tahmin Ortalama Adım Sayısı', fontsize=16)
plt.xlabel('Yöntemler', fontsize=12)
plt.ylabel('Ortalama Adım Sayısı', fontsize=12)

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

ai_accuracy = step_accuracy_rate('AI Prediction')
rule_based_accuracy = step_accuracy_rate('Rule-Based Prediction')
entropy_accuracy = step_accuracy_rate('Entropy Prediction')

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.lineplot(data=ai_accuracy, label="AI Accuracy", marker='o')
sns.lineplot(data=rule_based_accuracy, label="Rule-Based Accuracy", marker='o')
sns.lineplot(data=entropy_accuracy, label="Entropy Accuracy", marker='o')

plt.title('Adım Başına Doğru ve Yanlış Tahmin Oranı')
plt.xlabel('Adım Sayısı')
plt.ylabel('Başarı Yüzdesi')
plt.legend()
plt.grid(True)
plt.show()

# Heatmap için başarı oranları
step_accuracy_matrix = pd.DataFrame({
    'AI': ai_accuracy,
    'Rule-Based': rule_based_accuracy,
    'Entropy': entropy_accuracy
})
plt.figure(figsize=(10, 6))
sns.heatmap(step_accuracy_matrix.T, annot=True, cmap="Blues", fmt=".2f")
plt.title('Adım Başına Doğru Tahmin Oranı Heatmap')
plt.xlabel('Adım Sayısı')
plt.ylabel('Yöntem')
plt.show()
from wordcloud import WordCloud

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["AI Prediction"]])
wordcloud = WordCloud(width=800, height=400).generate(" ".join(all_predictions))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["Rule-Based Prediction"]])
wordcloud = WordCloud(width=800, height=400).generate(" ".join(all_predictions))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Her tahmin türü için kelime bulutu
all_predictions = pd.concat([data["Entropy Prediction"]])
wordcloud = WordCloud(width=800, height=400).generate(" ".join(all_predictions))

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

df = pd.concat([ai_best_words,rb_best_words,en_best_words])




