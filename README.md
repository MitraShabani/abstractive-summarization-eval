# Abstractive Summarization Evaluation for Scientific Papers

## Research Question
"Does domain-specific training produce more balanced section coverage
in abstractive summarization of scientific papers?"

## Models Compared
| Model | Type | Domain-specific | Long-doc |
|-------|------|-----------------|----------|
| pszemraj/led-base-book-summary | Abstractive | No | Yes |
| allenai/led-large-16384-arxiv | Abstractive | Yes | Yes |

Both models share the same long-document architecture (LED), isolating
domain-specific training as the single variable under study.

## Evaluation Metrics
### KL Divergence — KL(paper || summary)
Measures how much the summary's section distribution diverges from the
paper's section distribution. Zero sections are handled with an explicit penalty proportional to their
weight in the paper

### Entropy (balance of section coverage in summary)
Measures how evenly distributed the section labels are in the summary.
Higher entropy = more balanced coverage across all sections.

## Dataset
- 50 CS papers downloaded from arXiv
- Paper IDs used as filenames (e.g. 0807.1560.pdf)
- No reference summaries, evaluation is fully reference-free

## Pipeline
- **Step 1:** Extract plain text from PDFs using PyMuPDF
- **Step 2:** Generate summaries using both models via HuggingFace Transformers
- **Step 3:** Sentences are classified into four rhetorical categories:
Background, Methods, Results, Conclusion.
    - **Paper representation:** Abstracts fetched from the arXiv API are used
    as a proxy for the paper's section distribution. This avoids the
    unreliable classification of full paper body text, which lacks consistent
    IMRaD structure in CS papers.
    - **Summary classification:** Zero-shot classification using
    `facebook/bart-large-mnli` via Natural Language Inference (NLI).
- **Step 4:** Compute KL divergence and entropy:
Computed inside classify_sections.py immediately after classification.
Results saved per paper as JSON files.
- **Step 5:** Aggregate results and compare models statistically:
Aggregate KL divergence and entropy scores across all 50 papers.
Statistical comparison using Wilcoxon signed-rank test (paired,
non-parametric) since scores are paired per paper.

## Limitations

### Classifier reliability
The zero-shot classifier (bart-large-mnli) was not fine-tuned on
scientific text. Confidence scores are often below 0.5, meaning
classifications are uncertain. Several pretrained scientific sentence
classifiers were evaluated but found unreliable for CS paper text:
- allenai/scibert_scivocab_uncased: base model, no classification head
- gubartz/cls_scibert_abstruct: classifies everything as background
- ml4pubmed/scibert-scivocab-cased_pub_section: trained on biomedical,
  not CS papers

### Abstract as paper proxy
The paper's section distribution is derived from its abstract, not the
full body. This assumes abstracts faithfully represent the paper's
rhetorical structure. CS abstracts are free-form and may not always
cover all sections proportionally.

### LED-general output quality
The general LED model (pszemraj/led-base-book-summary) produced
occasional hallucinations and incoherent passages. Its KL divergence
scores reflect both poor structural faithfulness and poor output quality.
This is itself a finding — domain-specific training appears necessary
for coherent summarization of scientific papers.

### CS paper structure
CS papers do not follow strict IMRaD structure. Headers vary widely
across papers, making full-paper section classification unreliable.
This is why the abstract-based approach was adopted.

### Sample size
50 papers is a small sample. Findings should be interpreted as
preliminary and may not generalize across all CS subfields.

## Computation
- Runs on Google Colab (T4 GPU)
- Code stored on GitHub, outputs saved to Google Drive
- Sessions are resumable, both scripts skip already-processed papers

## Results

| Model | Mean KL Divergence | Std KL | Mean Entropy | Std Entropy |
|-------|-------------------|--------|--------------|-------------|
| LED-general | 0.2420 | 0.2123  | 0.9366  | 0.2439  |
| LED-arXiv | 0.1610 | 0.1396 | 0.9344  | 0.2065 |

### Statistical Test
Wilcoxon signed-rank test (paired, KL divergence):
- LED-general vs LED-arXiv
- statistic: 352.0, p-value: 0.0052
- **Significant at α=0.05**

### Interpretation
LED-arXiv (domain-specific) achieves significantly lower KL divergence
than LED-general (p=0.0052), indicating more balanced section coverage
that better mirrors the structural composition of the source papers.
Entropy scores are comparable between models, suggesting both produce
similarly balanced internal distributions — but LED-arXiv aligns more
faithfully with the paper's actual structure.

## Project Structure


├── papers/                50 arXiv PDFs (Google Drive only)

├── outputs/               generated data (Google Drive only)

│   ├── summaries/         one JSON per paper

│   └── distributions/     KL scores and distributions

├── results/               figures and statistics (committed)

│   ├── statistics.csv

│   ├── wilcoxon_test.txt

│   ├── kl_divergence_boxplot.png

│   ├── entropy_boxplot.png

│   └── section_distributions.png

├── generate_summaries.py   Step 2

├── classify_sections.py    Steps 3 and 4

├── analyze_results.py      Step 5

├── pdf_refine.py           text cleaning utilities

└── README.md