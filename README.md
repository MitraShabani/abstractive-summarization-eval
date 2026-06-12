# Abstractive Summarization Evaluation for Scientific Papers

## Research Question
"Does domain-specific training produce more balanced section coverage 
in abstractive summarization of scientific papers?"

## Models Compared
| Model | Type | Domain-specific | Long-doc |
|-------|------|-----------------|----------|
| allenai/led-base-16384 | Abstractive | No | Yes |
| allenai/led-large-16384-arxiv | Abstractive | Yes | Yes |

## Evaluation Metrics
- KL Divergence (summary vs paper section distribution)
- Entropy (balance of section coverage in summary)

## Pipeline
- **Step 1:** Extract plain text from PDFs using PyMuPDF
- **Step 2:** Generate summaries using all 3 models (HuggingFace Transformers)
- **Step 3:** Classify sentences by IMRaD section using CSABSTRUCT classifier
- **Step 4:** Compute KL divergence and entropy
- **Step 5:** Aggregate results and compare models statistically

## Project Structure