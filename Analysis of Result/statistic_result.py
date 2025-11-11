#!/usr/bin/env python3
# full_wordle_analysis.py
# Usage: python full_wordle_analysis.py results.csv

"""import sys
import os
from collections import defaultdict
from itertools import combinations
import math

# try to import packages, install if missing
def ensure_package(pkg, import_name=None):
    import importlib, subprocess, sys
    name = import_name if import_name else pkg
    try:
        return importlib.import_module(name)
    except Exception:
        print(f"Package '{pkg}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(name)

pd = ensure_package("pandas")
np = ensure_package("numpy")
plt = ensure_package("matplotlib.pyplot", "matplotlib.pyplot")
sns = ensure_package("seaborn")
scipy = ensure_package("scipy")
stats = scipy.stats
# posthoc and effect size libraries
pingouin = ensure_package("pingouin")
scikit_posthocs = ensure_package("scikit-posthocs", "scikit_posthocs")
statsmodels = ensure_package("statsmodels")
from statsmodels.stats.contingency_tables import mcnemar

# ------------------------------
# Parameters / Inputs
# ------------------------------
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "merged_result.csv"

OUTDIR = "analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------
# Read CSV
# ------------------------------
df = pd.read_csv(csv_path, dtype=str)  # read as str first
# Ensure Steps is numeric
if "Steps" not in df.columns:
    raise ValueError("CSV must contain a 'Steps' column.")
df["Steps"] = pd.to_numeric(df["Steps"], errors="coerce").astype(int)

methods = [
    "AI Prediction_Cosine",
    "Hybrid Prediction_Cosine",
    "Rule-Based Prediction",
    "AI Prediction_Euclidean",
    "Hybrid Prediction_Euclidean",
    "Entropy Prediction",
]
# validate methods exist
for m in methods:
    if m not in df.columns:
        raise ValueError(f"Method column missing in CSV: {m}")

# ------------------------------
# Build game-level evaluation DataFrame (first correct step per Target Word & method)
# ------------------------------
records = []
for target, group in df.groupby("Target Word"):
    # ensure group sorted by Steps ascending
    group_sorted = group.sort_values("Steps")
    for m in methods:
        correct_rows = group_sorted[group_sorted[m] == target]
        if len(correct_rows) > 0:
            first_step = int(correct_rows["Steps"].min())
            success = 1
        else:
            first_step = np.nan
            success = 0
        records.append({
            "Target Word": target,
            "Method": m,
            "Success": success,
            "First_Correct_Step": first_step
        })

eval_df = pd.DataFrame(records)
# pivot convenience tables
pivot_success = eval_df.pivot(index="Target Word", columns="Method", values="Success")
pivot_first_step = eval_df.pivot(index="Target Word", columns="Method", values="First_Correct_Step")

# ------------------------------
# 1) Accuracy (game-level) + 95% CI
# ------------------------------
def proportion_confint_mean(p, n, alpha=0.05):
    # normal approx CI for proportion
    z = stats.norm.ppf(1 - alpha/2)
    se = math.sqrt((p*(1-p))/n) if n > 0 else 0
    return max(0, p - z*se), min(1, p + z*se)

summary_rows = []
for m in methods:
    successes = pivot_success[m]
    n = len(successes)
    p = successes.mean()
    ci_low, ci_high = proportion_confint_mean(p, n)
    summary_rows.append({
        "Method": m,
        "Accuracy": p,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "N_games": n
    })
summary_df = pd.DataFrame(summary_rows).set_index("Method").sort_values("Accuracy", ascending=False)
summary_df.to_csv(os.path.join(OUTDIR, "summary_game_level.csv"))

# plot accuracy with CI
plt.figure(figsize=(10,5))
sns.barplot(x=summary_df.index, y="Accuracy", data=summary_df.reset_index(), errorbar=('ci', 95))
# add error bars (CI)
xs = range(len(summary_df))
for i, m in enumerate(summary_df.index):
    y = summary_df.loc[m, "Accuracy"]
    low = summary_df.loc[m, "CI_low"]
    high = summary_df.loc[m, "CI_high"]
    plt.errorbar(i, y, yerr=[[y-low], [high-y]], fmt='none', capsize=5)
plt.ylim(0,1.02)
plt.xticks(rotation=45, ha='right')
plt.title("Game-level Accuracy (95% CI)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "accuracy_ci.png"), dpi=300)
plt.close()

# ------------------------------
# 2) Average step to success (only games where success==1)
# ------------------------------
avg_steps = {}
for m in methods:
    steps = pivot_first_step[m].dropna().astype(int)
    avg_steps[m] = steps.mean() if len(steps)>0 else np.nan
avg_steps_df = pd.DataFrame.from_dict(avg_steps, orient='index', columns=["Avg_Step"])
avg_steps_df.to_csv(os.path.join(OUTDIR, "avg_steps.csv"))

plt.figure(figsize=(10,5))
sns.barplot(x=avg_steps_df.index, y="Avg_Step", data=avg_steps_df.reset_index(), errorbar=('ci', 95))
plt.xticks(rotation=45, ha='right')
plt.title("Average Step to First Correct Prediction (only where method succeeded)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "avg_steps.png"), dpi=300)
plt.close()

# ------------------------------
# 3) Step-wise success (first-correct proportions per step)
# ------------------------------
all_steps = sorted(df["Steps"].unique())
step_success = pd.DataFrame(index=all_steps, columns=methods).fillna(0.0)
for m in methods:
    series = pivot_first_step[m]
    counts = series.value_counts(dropna=True).to_dict()
    total_games = len(series)
    for s in all_steps:
        step_success.loc[s, m] = counts.get(s, 0) / total_games
step_success.to_csv(os.path.join(OUTDIR, "step_success_first_correct.csv"))

plt.figure(figsize=(10,6))
for m in methods:
    plt.plot(step_success.index, step_success[m], marker='o', label=m)
plt.xlabel("Step (first correct prediction step)")
plt.ylabel("Proportion of games solved at this step")
plt.title("Step-wise First-Correct Success Proportion")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "step_success_plot.png"), dpi=300)
plt.close()

# cumulative
cumulative = step_success.cumsum()
plt.figure(figsize=(10,6))
for m in methods:
    plt.plot(cumulative.index, cumulative[m], marker='o', label=m)
plt.xlabel("Step")
plt.ylabel("Cumulative proportion solved")
plt.title("Cumulative Success by Step")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cumulative_success.png"), dpi=300)
plt.close()

# ------------------------------
# 4) Similarity matrix (game-level success agreement)
# ------------------------------
sim = pd.DataFrame(index=methods, columns=methods, dtype=float)
for m1, m2 in combinations(methods, 2):
    # proportion of games where success result matches
    both = pivot_success[[m1, m2]].dropna()
    if len(both) == 0:
        val = np.nan
    else:
        val = (both[m1] == both[m2]).mean()
    sim.loc[m1, m2] = sim.loc[m2, m1] = val
for m in methods:
    sim.loc[m, m] = 1.0
sim.to_csv(os.path.join(OUTDIR, "similarity_matrix.csv"))

plt.figure(figsize=(8,6))
sns.heatmap(sim.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Game-level Success Agreement Between Methods")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "similarity_heatmap.png"), dpi=300)
plt.close()

# ------------------------------
# 5) Pairwise McNemar tests (binary success) with Bonferroni correction
# ------------------------------
pairs = []
pvals = []
for m1, m2 in combinations(methods, 2):
    table = pd.crosstab(pivot_success[m1], pivot_success[m2])
    # Ensure table is 2x2
    # Rows: m1 (0/1), Cols: m2 (0/1)
    table2 = table.reindex(index=[0,1], columns=[0,1], fill_value=0)
    # mcnemar expects [[a, b],[c, d]] -> b and c are discordant
    try:
        res = mcnemar(table2, exact=True)
        p = res.pvalue
    except Exception:
        # fallback to chi2 approx
        res = mcnemar(table2, exact=False)
        p = res.pvalue
    pairs.append((m1, m2))
    pvals.append(p)
# bonferroni
pvals_np = np.array(pvals)
bonf = np.minimum(pvals_np * len(pvals_np), 1.0)
mcnemar_df = pd.DataFrame({
    "pair": [f"{a} vs {b}" for a,b in pairs],
    "p_raw": pvals_np,
    "p_bonf": bonf
})
mcnemar_df.to_csv(os.path.join(OUTDIR, "pairwise_mcnemar.csv"), index=False)

# ------------------------------
# 6) Friedman test on First_Correct_Step (paired, uses ranks) + posthoc Nemenyi
# ------------------------------
# For Friedman we need a matrix of shape (n_games, n_methods) without NaNs.
# Strategy: replace NaN (no success) with max_step+1 (worse than any real step) -> treat as ordinal
max_step = int(df["Steps"].max())
filled_steps = pivot_first_step.fillna(max_step + 1).astype(int)
friedman_stat, friedman_p = stats.friedmanchisquare(*[filled_steps[m] for m in methods])
# posthoc nemenyi (scikit-posthocs expects numpy array with columns=methods)
posthoc = scikit_posthocs.posthoc_nemenyi_friedman(filled_steps.values)
posthoc_df = pd.DataFrame(posthoc, index=methods, columns=methods)
posthoc_df.to_csv(os.path.join(OUTDIR, "posthoc_nemenyi.csv"))

# ------------------------------
# 7) Effect sizes
#    - Cohen's d (paired) on First_Correct_Step: compute pairwise using only games where both methods succeeded OR use all with filled (max+1)
#    - Cliff's delta for binary success (pairwise)
# ------------------------------
eff_records = []
# Cohen's d (paired) using filled_steps (so includes failures as large value)
def cohens_d_paired(x, y):
    # x,y are arrays of same length
    d = (np.mean(x - y)) / (np.std(x - y, ddof=1) if np.std(x - y, ddof=1)!=0 else np.nan)
    return d

for m1, m2 in combinations(methods, 2):
    x = filled_steps[m1].values
    y = filled_steps[m2].values
    d = cohens_d_paired(x, y)
    # cliff's delta on binary success
    a = pivot_success[m1].values
    b = pivot_success[m2].values
    try:
        cliff = pingouin.compute_effsize(a, b, eftype="cliffs")
    except Exception:
        cliff = np.nan
    eff_records.append({
        "pair": f"{m1} vs {m2}",
        "cohens_d_paired": d,
        "cliffs_delta_binary": cliff
    })
eff_df = pd.DataFrame(eff_records)
eff_df.to_csv(os.path.join(OUTDIR, "effect_sizes.csv"), index=False)

# Also compute Cohen's d for each method vs baseline (smallest avg step)
baseline = avg_steps_df["Avg_Step"].astype(float).idxmin()
baseline_vals = filled_steps[baseline].values
baseline_effects = []
for m in methods:
    if m == baseline:
        continue
    d = cohens_d_paired(filled_steps[m].values, baseline_vals)
    baseline_effects.append({"Method_vs_baseline": f"{m} vs {baseline}", "cohens_d": d})
pd.DataFrame(baseline_effects).to_csv(os.path.join(OUTDIR, "effect_vs_baseline.csv"), index=False)

# ------------------------------
# 8) Post-hoc pairwise Wilcoxon on First_Correct_Step (paired) with Bonferroni correction
# ------------------------------
pairwise_w = []
pvals_w = []
for m1, m2 in combinations(methods, 2):
    x = filled_steps[m1].values
    y = filled_steps[m2].values
    try:
        stat, p = stats.wilcoxon(x, y)
    except Exception:
        # If all differences zero or other issue, set NaN
        stat, p = np.nan, np.nan
    pairwise_w.append((m1, m2))
    pvals_w.append(p)
pvals_w = np.array([np.nan_to_num(v, nan=1.0) for v in pvals_w])
pvals_w_bonf = np.minimum(pvals_w * len(pvals_w), 1.0)
pw_wilcoxon_df = pd.DataFrame({
    "pair": [f"{a} vs {b}" for a,b in pairwise_w],
    "p_raw": pvals_w,
    "p_bonf": pvals_w_bonf
})
pw_wilcoxon_df.to_csv(os.path.join(OUTDIR, "pairwise_wilcoxon_steps.csv"), index=False)

# ------------------------------
# 9) Generate results_discussion.txt (automated text)
# ------------------------------
def fmt_p(p):
    if np.isnan(p):
        return "NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"

with open(os.path.join(OUTDIR, "results_discussion.txt"), "w", encoding="utf-8") as f:
    f.write("Results and Discussion (auto-generated)\n")
    f.write("=====================================\n\n")
    f.write("1) Summary (Game-level Accuracy with 95% CI)\n")
    for m in summary_df.index:
        row = summary_df.loc[m]
        f.write(f"- {m}: Accuracy = {row['Accuracy']:.3f} (95% CI [{row['CI_low']:.3f}, {row['CI_high']:.3f}]), N={int(row['N_games'])}\n")
    f.write("\n2) Average Step to Success (only games where method succeeded)\n")
    for m in avg_steps_df.index:
        val = avg_steps_df.loc[m, "Avg_Step"]
        if np.isnan(val):
            f.write(f"- {m}: no successful games\n")
        else:
            f.write(f"- {m}: Avg Step = {val:.2f}\n")
    f.write("\n3) Agreement between methods (game-level)\n")
    f.write(f"- Similarity (agreement) matrix saved as 'similarity_matrix.csv'\n\n")
    f.write("4) Statistical tests\n")
    f.write(f"- Friedman test on first-correct step (failures treated as step {max_step+1}): Chi2 = {friedman_stat:.4f}, {fmt_p(friedman_p)}\n")
    significant_pairs = pw_wilcoxon_df[pw_wilcoxon_df["p_bonf"] < 0.05]
    if len(significant_pairs) > 0:
        f.write("- Pairwise Wilcoxon (paired) post-hoc (First_Correct_Step) with Bonferroni correction found significant differences in:\n")
        for _, r in significant_pairs.iterrows():
            f.write(f"  * {r['pair']}: {fmt_p(r['p_bonf'])}\n")
    else:
        f.write("- No pairwise Wilcoxon comparisons reached p < 0.05 after Bonferroni correction.\n")
    f.write("\n- Pairwise McNemar tests (binary success) with Bonferroni correction:\n")
    small = mcnemar_df.sort_values("p_bonf").head(10)
    for _, r in small.iterrows():
        f.write(f"  * {r['pair']}: p_raw = {r['p_raw']:.4f}, p_bonf = {r['p_bonf']:.4f}\n")
    f.write("\n5) Effect sizes\n")
    f.write("- Pairwise Cohen's d (paired) on step values and Cliff's delta for binary success:\n")
    for _, r in eff_df.iterrows():
        f.write(f"  * {r['pair']}: Cohen's d (paired) = {np.nan_to_num(r['cohens_d_paired']):.3f}, Cliff's delta = {np.nan_to_num(r['cliffs_delta_binary']):.3f}\n")
    f.write("\n6) Practical interpretation suggestions (auto)\n")
    # simple heuristic
    best = summary_df.index[0]
    f.write(f"- The top-performing method by accuracy is {best} (Accuracy = {summary_df.loc[best,'Accuracy']:.3f}).\n")
    f.write("- If Friedman test is significant, prefer reporting post-hoc pairwise results and effect sizes to quantify magnitude.\n")
    f.write("- Discuss failure modes using the 'step_success_first_correct.csv' (distribution of first-correct steps) and error examples.\n")
    f.write("\nFiles produced:\n")
    for fname in sorted(os.listdir(OUTDIR)):
        f.write(f"- {fname}\n")

# ------------------------------
# 10) Save DataFrames for inspection
# ------------------------------
summary_df.to_csv(os.path.join(OUTDIR, "summary_game_level.csv"))
pivot_success.to_csv(os.path.join(OUTDIR, "pivot_success_per_game.csv"))
pivot_first_step.to_csv(os.path.join(OUTDIR, "pivot_first_step_per_game.csv"))

print("Analysis complete. Outputs saved to folder:", OUTDIR)
print("Key files: summary_game_level.csv, effect_sizes.csv, pairwise_mcnemar.csv, posthoc_nemenyi.csv, results_discussion.txt")"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import kruskal, wilcoxon, spearmanr
import scikit_posthocs as sp

# ===========================================
# 🔹 1. Veri Yükleme
# ===========================================
df = pd.read_csv("../merged_result.csv")
methods = [
    "Rule-Based Prediction",
    "Hybrid Prediction_Euclidean",
    "Entropy Prediction",
]

# =========================
# 🔹 Bar Chart - Top 5 Kelime
# =========================
rows = []
for m in methods:
    total = len(df[m])  # yöntemin toplam tahmin sayısı
    top_counts = df[m].value_counts().head(5)
    for word, count in top_counts.items():
        freq = count / total
        rows.append({
            "WORD": word,
            "METHOD": m,
            "COUNT": count,
            "Frequency": round(freq, 4)
        })

top_freq_df = pd.DataFrame(rows)
top_freq_df.to_csv("top5_kelime_frekans.csv", index=False, encoding="utf-8-sig")
print(top_freq_df)

# ===========================================
# 🔹 2. Her oyun için ilk doğru tahmin stepini bulma
# ===========================================
results = []
for m in methods:
    for target, group in df.groupby("Target Word"):
        correct_rows = group[group[m] == target]
        if len(correct_rows) > 0:
            first_step = correct_rows["Steps"].min()
            success = 1
        else:
            first_step = None
            success = 0
        results.append({
            "Target Word": target,
            "Method": m,
            "Success": success,
            "First_Correct_Step": first_step
        })

eval_df = pd.DataFrame(results)

# ===========================================
# 🔹 3. Oyun Bazlı Yöntemlerin Accuracy ve Ortalama Step
# ===========================================
accuracy = eval_df.groupby("Method")["Success"].mean()
avg_step = eval_df[eval_df["Success"]==1].groupby("Method")["First_Correct_Step"].mean()
summary = pd.DataFrame({"Accuracy": accuracy, "Average Step": avg_step})
summary.to_csv("game_level_summary.csv")
print("\n--- Game-level Summary ---\n", summary)

# ===========================================
# 🔹 4. Step Bazlı Başarı Oranı
# ===========================================
step_success = defaultdict(dict)
for m in methods:
    method_data = eval_df[eval_df["Method"]==m]
    for step in sorted(df["Steps"].unique()):
        ratio = (method_data["First_Correct_Step"]==step).mean()
        step_success[m][step] = ratio
step_success_df = pd.DataFrame(step_success).fillna(0)
markers = ['o', 'D', '*']
plt.figure(figsize=(8,5))
for m, marker in zip(methods, markers):
    plt.plot(step_success_df.index, step_success_df[m], marker=marker, label=m)
plt.title("Step-wise Success Rate")
plt.xlabel("Step")
plt.ylabel("Proportion of Games Solved")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("step_success_rate.png", dpi=300)
plt.show()

# Kümülatif başarıyı hesapla
cumulative_df = step_success_df.cumsum()

plt.figure(figsize=(8,5))
for m, marker in zip(cumulative_df.columns, markers):
    plt.plot(cumulative_df.index, cumulative_df[m], marker=marker, label=m)

plt.title("Cumulative Step-wise Success Rate")
plt.xlabel("Step")
plt.ylabel("Cumulative Proportion of Games Solved")
plt.xticks(cumulative_df.index)  # x-axis adım sayıları
plt.ylim(0, 1.05)  # Yüzdeyi 0-100% aralığında göstermek için
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Method")
plt.tight_layout()
plt.savefig("cumulative_success_rate.png", dpi=300)
plt.show()

# ===========================================
# 🔹 5. Kruskal-Wallis Test (3+ yöntem karşılaştırma)
# Farklı yöntemlerin başarı oranları arasında istatistiksel olarak anlamlı bir fark olup olmadığını ölçe
# ===========================================
success_lists = [eval_df[eval_df["Method"]==m]["Success"] for m in methods]
stat, p = kruskal(*success_lists)
print(f"\nKruskal-Wallis test: stat={stat:.4f}, p={p:.5f}")

# ===========================================
# 🔹 6. Post-hoc Dunn Test (hangi çiftler farklı)
# Hangi iki yöntem arasında fark var ve arkın anlamlılık düzeylerini hesaplar
# ===========================================
data_wide = eval_df.pivot(index="Target Word", columns="Method", values="Success")
dunn_results = sp.posthoc_dunn(
    eval_df,
    val_col="Success",
    group_col="Method",
    p_adjust="bonferroni"
)
dunn_results.to_csv("dunn_posthoc.csv")
print("\nDunn post-hoc test results saved to 'dunn_posthoc.csv'")

# ===========================================
# 🔹 7. Etki Büyüklüğü (Cliff’s Delta) - Nonparametrik
#Cliff’s Delta  “farkın büyüklüğü / yönü nedir” i bulmak için kullanılır
# ===========================================
def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    more = sum([1 for xi in x for yi in y if xi > yi])
    less = sum([1 for xi in x for yi in y if xi < yi])
    delta = (more - less) / (n_x * n_y)
    return delta

effect_size = pd.DataFrame(index=methods, columns=methods)
for i in range(len(methods)):
    for j in range(i+1, len(methods)):
        x = data_wide[methods[i]].dropna()
        y = data_wide[methods[j]].dropna()
        delta = cliffs_delta(x, y)
        effect_size.loc[methods[i], methods[j]] = delta
        effect_size.loc[methods[j], methods[i]] = -delta
for m in methods:
    effect_size.loc[m,m] = 0
effect_size.to_csv("cliffs_delta_matrix.csv")
print("\nCliff's Delta effect size matrix saved to 'cliffs_delta_matrix.csv'")

# ===========================================
# 🔹 8. Yöntemler Arası Benzerlik
# ===========================================
sim_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
for i in methods:
    for j in methods:
        sim_matrix.loc[i,j] = (data_wide[i]==data_wide[j]).mean()
plt.figure(figsize=(7,6))
sns.heatmap(sim_matrix.astype(float), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Game-level Success Similarity Between Methods")
plt.tight_layout()
plt.savefig("method_similarity_heatmap.png", dpi=300)
plt.show()
sim_matrix.to_csv("method_similarity_matrix.csv")

print("\n✅ All statistical results computed and saved!")
