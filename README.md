# Abstractive Summarization Evaluation for Scientific Papers

## Research Question
"Does domain-specific training and long-document
architecture improve structural faithfulness in
abstractive summarization of scientific papers?"

## Models Compared
| Model | Type | Domain-specific | Long-doc |
|-------|------|-----------------|----------|
| BART | Abstractive | No | No |
| LED | Abstractive | No | Yes |
| allenai/led-large-16384-arxiv | Abstractive | Yes | Yes |

## Evaluation Framework
Structural faithfulness is measured using KL divergence between 
the section-label distribution of the original paper and the 
generated summary. Sentences are classified into IMRaD categories 
(Background, Methods, Results, Conclusion) using a pretrained 
CS-domain sentence classifier (CSABSTRUCT). A lower KL divergence 
indicates the summary more faithfully mirrors the structural composition of the source paper.

This evaluation approach is adapted from a prior independent project 
on extractive summarization:
 https://github.com/MitraShabani/extractive-summarization-eval.git

## Pipeline
- **Step 1:** Extract plain text from PDFs using PyMuPDF
- **Step 2:** Generate summaries using all 3 models (HuggingFace Transformers)
- **Step 3:** Classify sentences by IMRaD section using CSABSTRUCT classifier
- **Step 4:** Compute KL divergence between paper and summary distributions
- **Step 5:** Aggregate results and compare models statistically

## Project Structure