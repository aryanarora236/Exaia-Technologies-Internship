# Explainable AI & Text Analytics — Mental Health Disorder Classification
 
> Research internship at **Exaia Technologies** focused on detecting mental health disorders from text using NLP and machine learning, with a deep dive into synthetic data generation and its effect on downstream model performance.
 
---
 
## Overview
 
Manually labeled mental health datasets are scarce and expensive to produce. This project explores whether high-quality **synthetic data** can supplement real training data without degrading — and potentially improving — classifier performance.
 
The work covers the full pipeline: disorder detection modeling, synthetic data generation across 8 controlled settings, fidelity analysis of the generated data, and rigorous downstream evaluation.
 
---
 
## Research Questions
 
1. Can NLP/ML models reliably classify mental health disorders from unstructured text?
2. How realistic is synthetically generated mental health text across different generation strategies?
3. Does adding synthetic data to training sets improve, hurt, or have no effect on model performance?
---
 
## What's in this repo
 
```
Exaia-Technologies-Internship/
├── Data Processing Scripts/              # Cleaning, tokenization, feature engineering
├── Synthetic Control Data Generation Scripts/   # Scripts for generating control (negative) samples
├── Synthetic Positive Data Generation Scripts/  # Scripts for generating disorder-positive samples
├── Final Datasets/                       # Processed datasets used in final experiments
├── Final Data Analysis Results/          # Model evaluation outputs, metrics, plots
├── Testing Scripts/                      # Validation and sanity-check scripts
├── FinalReport.pdf                       # Full write-up: methodology, results, discussion
└── ResearchSymposiumPoster.pdf           # Poster presented at research symposium
```
 
---
 
## Methods
 
### Disorder Detection
- Text preprocessing: tokenization, stopword removal, lemmatization
- Feature extraction: TF-IDF, sentence embeddings
- Classifiers: logistic regression, SVM, fine-tuned transformer models
### Synthetic Data Generation (8 settings)
Generation strategies varied along two axes:
- **Prompting strategy** — zero-shot, few-shot, persona-conditioned
- **Model type** — GPT-family, open-weight LLMs
### Fidelity & Realism Analysis
- Statistical similarity between real and synthetic distributions
- Human evaluation (lexical diversity, coherence, disorder-signal retention)
- Embedding-space overlap (cosine similarity, MMD)
### Downstream Evaluation
- Trained classifiers on: real-only, synthetic-only, and mixed datasets
- Measured accuracy, F1, and AUC across all 8 synthetic settings
- Identified which generation strategies most improved model robustness
---
 
## Key Findings
 
See [`FinalReport.pdf`](./FinalReport.pdf) and [`ResearchSymposiumPoster.pdf`](./ResearchSymposiumPoster.pdf) for full results.
 
High-level takeaways:
- Certain synthetic generation settings meaningfully improved classifier F1 on held-out real data
- Persona-conditioned generation produced the most realistic text by embedding-space overlap
- Zero-shot synthetic data alone was insufficient — mixing with real data was essential
---
 
## Stack
 
Python · scikit-learn · HuggingFace Transformers · pandas · NumPy · matplotlib
 
---
 
## Context
 
This work was completed as part of a research internship at Exaia Technologies. The final findings were presented at an internal research symposium.
