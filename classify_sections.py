import json
import spacy
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from transformers import pipeline
from scipy.stats import entropy
import numpy as np

""" Based on some limitations I changed the logic to:
. fetch abstract from arXiv API (with using paper ID)
. classify abstract sentences -> paper's section distributio
. classify summary sentences -> summary's section distribution
. compute KL divergence and Entropy """

# --------------------------------------------
# Configuration
# --------------------------------------------
SUMMARIES_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/summaries")
OUTPUT_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/distributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["background and introduction", "methodology and methods", "results and findings", "conclusion"]
LABEL_MAP = {
    "background and introduction": "background",
    "methodology and methods": "methods",
    "results and findings": "results",
    "conclusion": "conclusion"
}
SECTIONS = list(LABEL_MAP.values())

# --------------------------------------------
# Load models once
# --------------------------------------------
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print("Loading zero-shot classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

# --------------------------------------------
# Fetch abstract from arXiv API
# --------------------------------------------
def fetch_abstract(paper_id):
    url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", namespace)
        if entry is None:
            print(f"  WARNING: No entry found for {paper_id}")
            return ""
        abstract = entry.find("atom:summary", namespace).text.strip()
        return abstract
    except Exception as e:
        print(f"  ERROR fetching abstract for {paper_id}: {e}")
        return ""

# --------------------------------------------
# Split text into sentences using spaCy
# --------------------------------------------
def split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 5]
    return sentences

# --------------------------------------------
# Classify sentences and return section distribution
# --------------------------------------------
def get_distribution(sentences):
    if not sentences:
        return {section: 0.0 for section in SECTIONS}

    results = classifier(
        sentences,
        candidate_labels=LABELS,
        batch_size=8
    )

    # Handle single sentence case — classifier returns dict not list
    if isinstance(results, dict):
        results = [results]

    counts = {section: 0 for section in SECTIONS}
    for result in results:
        top_label = result["labels"][0]
        short_label = LABEL_MAP[top_label]
        counts[short_label] += 1

    total = sum(counts.values())
    distribution = {section: count / total for section, count in counts.items()}
    return distribution

# --------------------------------------------
# Compute KL divergence with Penalty
# --------------------------------------------
def compute_kl(paper_dist, summary_dist):

    kl = 0.0
    for section in SECTIONS:
        expected = paper_dist[section]  # paper's distribution
        actual = summary_dist[section]  # summary's distribution
        
        if actual > 0:
            # Standard KL term for covered sections, we add nothing
            kl += actual * np.log(actual / expected) if expected > 0 else 0
        else:
            # Penalty for completely ignored sections
            kl += expected
    
    return float(kl)

# --------------------------------------------
# Compute entropy
# --------------------------------------------
def compute_entropy(dist):
    smooth = 1e-10
    p = np.array([dist[s] + smooth for s in SECTIONS])
    p = p / p.sum()
    return float(entropy(p))

# --------------------------------------------
# Main loop
# --------------------------------------------
json_files = sorted(SUMMARIES_DIR.glob("*.json"))
print(f"Found {len(json_files)} summary files")

for json_path in json_files:
    paper_id = json_path.stem
    output_file = OUTPUT_DIR / f"{paper_id}.json"

    if output_file.exists():
        print(f"Skipping {paper_id} — already done")
        continue

    print(f"\nProcessing: {paper_id}")

    # Load summaries
    with open(json_path) as f:
        data = json.load(f)

    # Fetch abstract
    print("  Fetching abstract from arXiv...")
    abstract = fetch_abstract(paper_id)
    if not abstract:
        print(f"  Skipping {paper_id} — no abstract found")
        continue

    # Get paper distribution from abstract
    print("  Classifying abstract sentences...")
    abstract_sentences = split_sentences(abstract)
    print(f"  Found {len(abstract_sentences)} sentences in abstract")
    paper_dist = get_distribution(abstract_sentences)
    print(f"  Paper distribution: {paper_dist}")

    result = {
        "paper_id": paper_id,
        "abstract": abstract,
        "paper_distribution": paper_dist,
        "paper_entropy": compute_entropy(paper_dist),
        "summaries": {}
    }

    # Process each model's summary
    for model_key, summary in data["summaries"].items():
        if not summary:
            print(f"  Skipping {model_key} — no summary")
            result["summaries"][model_key] = None
            continue

        print(f"  Classifying {model_key} summary...")
        summary_sentences = split_sentences(summary)
        print(f"  Found {len(summary_sentences)} sentences in summary")
        summary_dist = get_distribution(summary_sentences)
        kl = compute_kl(paper_dist, summary_dist)
        ent = compute_entropy(summary_dist)

        result["summaries"][model_key] = {
            "distribution": summary_dist,
            "entropy": ent,
            "kl_divergence": kl
        }

        print(f"  {model_key} → KL: {kl:.4f}, Entropy: {ent:.4f}")

    Path(output_file).write_text(json.dumps(result, indent=2))
    print(f"  Saved to {output_file}")

print("\nAll done!")