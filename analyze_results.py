from pathlib import Path
import json
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
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
paper_dists = []
model_dists = {} # {model_key: [dist1, dist2, ...]}

for j in sorted(DISTRIBUTIONS_DIR.glob("*.json")):
    data = json.loads(Path(j).read_text())

    paper_id = data["paper_id"]
    paper_entropy = data["paper_entropy"]
    summaries = data["summaries"]

    # Store paper distribution
    paper_dists.append(data["paper_distribution"])

    for model_key, summary_data in summaries.items():
        results.append({
            "paper_id": paper_id,
            "model": model_key,
            "kl_divergence": summary_data["kl_divergence"],
            "entropy": summary_data["entropy"],
            "paper_entropy": paper_entropy
        })

        # Store model distribution
        if model_key not in model_dists:
            model_dists[model_key] = []
        model_dists[model_key].append(summary_data["distribution"])

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
kl_data = [df[df["model"] == model]["kl_divergence"].values for model in models]

ax.boxplot(kl_data, labels=models)
ax.set_title("KL Divergence by Model")
ax.set_xlabel("Model")
ax.set_ylabel("KL Divergence (lower = more faithful)")

plt.suptitle("") # replaces that automatic title with an empty string
plt.tight_layout()
plt.savefig(RESULTS_DIR / "kl_divergence_boxplot.png", dpi=150)
plt.close()
print("\nSaved Figure 1: KL divergence box plot")

# --------------------------------------------
# Figure 2 — Entropy box plot
# --------------------------------------------
fig, ax = plt.subplots()
entropy_data = [df[df["model"] == model]["entropy"].values for model in models]

ax.boxplot(entropy_data, labels=models)
ax.set_title("Entropy by Model")
ax.set_xlabel("Model")
ax.set_ylabel("Entropy (higher = more balanced)")

plt.suptitle("")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "entropy_boxplot.png", dpi=150)
plt.close()
print("Saved Figure 2: Entropy box plot")

# --------------------------------------------
# Figure 3 — Average section distributions
# --------------------------------------------
SECTIONS = ["background", "methods", "results", "conclusion"]

# average value across 50 papers
paper_mean = {section: np.mean([d[section] for d in paper_dists]) for section in SECTIONS}
# average value for each model across 50 papers
model_means = {model: {section: np.mean([d[section] for d in dists]) for section in SECTIONS} 
               for model, dists in model_dists.items()}

# Plot
x = np.arange(len(SECTIONS)) # one position for each section which is [0,1,2,3]
width = 0.25 # each bar
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width,
    [paper_mean[section] for section in SECTIONS],
    width,
    label="Paper (abstract)",
    color="gray"
    )

for i, (model, dist) in enumerate(model_means.items()):
    ax.bar(
        x + i*width, # i is used to shift each model's bars horizontally so they don't overlap
        [dist[section] for section in SECTIONS],
        width,
        label=model
        )

ax.set_xticks(x)
ax.set_xticklabels(SECTIONS)
ax.set_ylabel("Proportion")
ax.set_title("Average Section Distribution: Paper(abstract) vs Summaries")
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "section_distributions.png", dpi=150)
plt.close()
print("Saved Figure 3: Section distributions")
