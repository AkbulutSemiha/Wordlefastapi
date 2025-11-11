import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd


df = pd.read_csv("../model/colab_5fold_results/training_report.csv")
# 2️⃣ En iyi epoch (Test Accuracy bazlı)
best_epoch =df.loc[df["Test Acc"].idxmax(), ["Train Loss","Train Acc","Test Loss","Test Acc"]]
print("\n=== Best Fold ===")
print(best_epoch)

best_per_fold = df.loc[df.groupby("Fold")["Test Acc"].idxmax()].reset_index(drop=True)

# Sonuçları göster
print(best_per_fold)

# 3️⃣ Fold bazlı istatistikler (bu örnekte tek fold var, birden fazla fold olursa groupby ile)
fold_stats = df.groupby("Fold")[["Train Loss","Train Acc","Test Loss","Test Acc"]].agg(['mean','std'])
print("\n=== Fold Bazlı İstatistikler ===")
fold_stats.to_csv("foldbazlıortalamatandartsapma.csv")
print(fold_stats)

# 4️⃣ Öğrenme eğrileri görselleştirme
plt.figure(figsize=(10,5))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
plt.plot(df["Epoch"], df["Test Loss"], label="Test Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Epoch"], df["Train Acc"], label="Train Accuracy", marker='o')
plt.plot(df["Epoch"], df["Test Acc"], label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# 5️⃣ Anlamlılık testi: Paired t-test (Train vs Test Accuracy)
t_stat, p_value = stats.ttest_rel(df["Train Acc"], df["Test Acc"])
print("\n=== Paired t-test (Train Acc vs Test Acc) ===")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
