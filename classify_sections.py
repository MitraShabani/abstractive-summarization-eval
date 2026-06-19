# --------------------------------------------
# Import dependencies
# --------------------------------------------

import json
import os
import spacy
from transformers import pipeline
from pathlib import Path
import fitz
import sys
from pdf_refine import clean_text

# --------------------------------------------
# Configuration
# --------------------------------------------

SUMMARIES_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/summaries")
PAPERS_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/papers")
OUTPUT_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/distributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["background and introduction", "methodology and methods", "results and findings", "conclusion"]
LABEL_MAP = {
    "background and introduction": "background",
    "methodology and methods": "methods",
    "results and findings": "results",
    "conclusion": "conclusion"
}

# --------------------------------------------
# Load models once
# --------------------------------------------

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

""" I'm using a zero-shot classifier model (bart-large-mnli) which puts labels as hypothesis,
calculates an entailment probability and picks the highest one as true label."""

print("Loading zero-shot classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # GPU
)

# --------------------------------------------
#  Extract Original Paper Text from PDF
# --------------------------------------------

def get_paper_text(paper_id):
    """ Read the original paper text from Drive to compute the paper's section distribution."""

    sys.path.append("/content/abstractive-summarization-eval")

    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        print(f"  WARNING: PDF not found for {paper_id}")
        return ""

    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text()
    return clean_text(full_text)


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
        return {label: 0.0 for label in LABEL_MAP.values()}

    # Batch classify
    """ Instead of running the NLI model once per sentence, we run it on 8 sentences at a time. 
    The GPU processes them in parallel"""

    results = classifier(
        sentences,
        candidate_labels=LABELS,
        batch_size=8
    )

    """ Classifier returns the labels sorted by confidence score.
    [0] takes the first item, the label with the highest confidence score."""

    # Count labels
    counts = {label: 0 for label in LABEL_MAP.values()}
    for result in results:
        highest_scored_label = result["labels"][0]
        Shortened_label = LABEL_MAP[highest_scored_label]
        counts[Shortened_label] += 1

    # Convert to distribution
    total = sum(counts.values())
    distribution = {label: count / total for label, count in counts.items()}
    return distribution


# --------------------------------------------
# Main loop
# --------------------------------------------

json_files = sorted(SUMMARIES_DIR.glob("*.json"))
print(f"Found {len(json_files)} summary files")

for json_path in json_files:
    paper_id = json_path.stem
    output_file = OUTPUT_DIR / f"{paper_id}.json"

    # Skip if already processed
    if output_file.exists():
        print(f"Skipping {paper_id} — already done")
        continue

    print(f"\nProcessing: {paper_id}")

    # Load summaries
    with open(json_path) as f:
        data = json.load(f)

    result = {"paper_id": paper_id, "distributions": {}}

    # Paper distribution
    print("  Classifying paper sentences...")
    paper_text = get_paper_text(paper_id)
    paper_sentences = split_sentences(paper_text)
    print(f"  Found {len(paper_sentences)} sentences in paper")
    result["distributions"]["paper"] = get_distribution(paper_sentences)

    # Summary distributions
    for model_key, summary in data["summaries"].items():
        if not summary:
            print(f"  Skipping {model_key} — no summary")
            result["distributions"][model_key] = None
            continue

        print(f"  Classifying {model_key} summary...")
        summary_sentences = split_sentences(summary)
        print(f"  Found {len(summary_sentences)} sentences in summary")
        result["distributions"][model_key] = get_distribution(summary_sentences)

    # Save
    Path(output_file).write_text(json.dumps(result, indent=2))
    print(f"  Saved to {output_file}")

print("\nAll done!")