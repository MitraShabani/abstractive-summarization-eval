from pathlib import Path
import json
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
# --------------------------------------------
# Configuration
# --------------------------------------------
DISTRIBUTIONS_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/distributions")
RESULTS_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------
# Load all results
# --------------------------------------------
results = []
for j in sorted(DISTRIBUTIONS_DIR.glob("*.json")):
    data = json.loads(Path(j).read_text())

    paper_id = data["paper_id"]
    paper_entropy = data["paper_entropy"]
    summaries = data["summaries"]

    for model_key, summary_data in summaries.items():
        results.append({
            "paper_id": paper_id,
            "model": model_key,
            "kl_divergence": summary_data["kl_divergence"],
            "entropy": summary_data["entropy"],
            "paper_entropy": paper_entropy
        })

df = pd.DataFrame(results)
print(f"Loaded {len(df)} results")
print(df.groupby("model")[["kl_divergence", "entropy"]].describe()) # quick statistical summary

# --------------------------------------------
# Statistics table
# --------------------------------------------
statisrics_table = df.groupby("model")[["kl_divergence", "entropy"]].agg(["mean", "std"]).round(4)
statisrics_table.to_csv(RESULTS_DIR / "statistics.csv")
print("")

# --------------------------------------------
# Wilcoxon
# --------------------------------------------
""" is the difference in KL scores between the two models real, or could it be due to random chance? """

models = df["model"].unique()
if len(models) == 2:
    model_a, model_b = models[0], models[1]

    # Get paired scores and reshape the dataframe so each row is one paper (each paper, different models)
    paired = df.pivot(index="paper_id", columns="model", values="kl_divergence").dropna()

    # stat -> the test statistic p_value ->  the probability that this difference occurred by chance
    stat, p_value = wilcoxon(paired[model_a], paired[model_b])

    (RESULTS_DIR / "wilcoxon_test.txt").write_text(
        f"Wilcoxon signed-rank test (KL divergence)\n"
        f"{model_a} vs {model_b}\n"
        f"statistic: {stat:.4f}, p-value: {p_value:.4f}\n"
        f"{'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05\n"
    )
    print("Wilcoxon signed-rank test is done.")

# --------------------------------------------
# Figure 1 — KL divergence box plot
# --------------------------------------------
fig, ax = plt.subplots()
df.boxplot(column= "KL_Divergence", by="model")

ax.set_title("KL Divergence by Model")
ax.set_xlabel("Model")
ax.set_ylabel("KL Divergence")

plt.savefig(RESULTS_DIR / "kl_divergence_boxplot.png", dpi=150)
plt.close()
print("\nSaved Figure 1: KL divergence box plot")

# --------------------------------------------
# Figure 2 — Entropy box plot
# --------------------------------------------
fig, ax = plt.subplots()
df.boxplot(column="entropy", by="model", ax=ax)

ax.set_title("Entropy by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Entropy (higher = more balanced)")

plt.suptitle("")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "entropy_boxplot.png", dpi=150)
plt.close()
print("Saved Figure 2: Entropy box plot")

print("\nAll results saved to", RESULTS_DIR)

