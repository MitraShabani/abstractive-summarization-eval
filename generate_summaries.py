# --------------------------------------------
# Mount Google Drive
# --------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Colab — skipping Drive mount")

# --------------------------------------------
# Clone GitHub repo
# --------------------------------------------
import subprocess
subprocess.run(["git", "clone",
    "https://github.com/MitraShabani/abstractive-summarization-eval.git",
    "/content/abstractive-summarization-eval"])

# --------------------------------------------
# Install dependencies
# --------------------------------------------
""" installs libraries onto the machine,
same as installing them by terminal"""

subprocess.run(["pip", "install", "-q",
    "transformers", "torch", "accelerate",
    "pymupdf", "sentencepiece"])

# --------------------------------------------
# Import dependencies
# --------------------------------------------
import json
import torch
import pymupdf # fitz
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf_refine import clean_text

# --------------------------------------------
# Configuration
# --------------------------------------------
PAPERS_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/papers")
OUTPUT_DIR = Path("/content/drive/MyDrive/abstractive-summarization-eval/outputs/summaries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "bart": "facebook/bart-large-cnn",
    "led": "allenai/led-base-16384",
    "led_arxiv": "allenai/led-large-16384-arxiv"
}
# 0 -> GPU , -1 -> CPU
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")

# --------------------------------------------
# Extract text from PDF
# --------------------------------------------
def extract_text_from_pdf(pdf_path):

    # the input for model (full_text) must be plain string.
    full_text = " "

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text()
    return clean_text(full_text)

# --------------------------------------------
# Summarization
# --------------------------------------------
def summarization (text, model_key, model_value):

    # Token limits differ per model, for BART 1024 tukens of input and 16384 for LED :
    if model_key == "bart":
        max_input = 1024
        max_output = 256  # how long the summary can be
        min_output = 64   # stops the model from producing a one-sentence summary
    else:
        max_input = 16384
        max_output = 512
        min_output = 128

    # models expects the string format so we define 'device' as a string and call it for inputs and the model.
    # It garantees they are always on the same device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # conversion, text -> numbers
    tokenizer = AutoTokenizer.from_pretrained(model_value)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input).to(device)

    """ encoder takes those numbers and transform them into a rich context-aware matrix.
        (AutoModelForSeq2SeqLM adds a decoder head that is specifically designed
        to generate output sequences token by token, which is what summarization requires)
        decoder takes the matrix and generates the output one word at a time, but still a sequence of token numbers.
    """

    # encoder and decoder run here
    model = AutoModelForSeq2SeqLM.from_pretrained(model_value).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_output,
        min_new_tokens=min_output,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
        no_repeat_ngram_size=4, # prevents exact phrase repetition
        repetition_penalty=2.0 # discourages repeating any individual word too often
    )

    # Decode tokens back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --------------------------------------------
# Main loop
# --------------------------------------------
papers = sorted(PAPERS_DIR.glob("*.pdf"))[:2]
print(f"{len(papers)} PDFs found")

for paper in papers:

    paper_id = paper.stem
    output_file = OUTPUT_DIR / f"{paper_id}.json"

    # In case of repetition
    if output_file.exists():
        print(f"Skipping {paper_id}: has already done")
        continue

    print(f"\nProcessing: {paper_id}")

    full_text = extract_text_from_pdf(paper)
    summaries = {"paper_id": paper_id, "summaries": {}}

    for model_key, model_value in MODELS.items():
        try:
            summary = summarization(full_text, model_key, model_value)
            summaries["summaries"][model_key] = summary
        except Exception as e:
            print(f" ERROR with {model_key} on {paper_id}: {e}")
            summaries["summaries"][model_key] = None

     # Save immediately to GoogleDrive
    with open(output_file, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved to {output_file}")

print("\nAll done!")
