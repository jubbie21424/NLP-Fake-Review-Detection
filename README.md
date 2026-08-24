# Detecting Deceptive Reviews: A Comparative NLP Study

Comparing a TF-IDF + Logistic Regression baseline against a fine-tuned BERT classifier for detecting computer-generated fake product reviews, with a subgroup error analysis across review length and star rating.


---

## Project Highlights

- **Two modelling approaches compared head-to-head.** A TF-IDF + Logistic Regression baseline (with unigram, bigram, metadata, and combined variants) is benchmarked against a fine-tuned `bert-base-uncased` classifier on the same held-out test set.
- **BERT delivers the strongest result.** BERT reaches an F1-score of 0.928 (accuracy 92.6%, precision 0.897, recall 0.962), outperforming the best Logistic Regression benchmark (F1 = 0.885) while keeping precision and recall well balanced.
- **40,493 reviews, near-perfectly balanced.** Built on the `theArijitDas/Fake-Reviews-Dataset` (Hugging Face), cleaned and deduplicated from 40,526 raw reviews, with a near 50/50 split between genuine and computer-generated reviews.
- **Goes beyond aggregate accuracy.** Row-level predictions are analysed by review length, star rating, and their interaction, revealing *where* and *why* each model fails — not just how often.
- **Review length is the single strongest driver of error.** Short reviews are misclassified roughly 4x more often than long reviews under both models (Logistic Regression: 19.5% vs 4.5%; BERT: 13.5% vs 1.9%), pointing to textual sparsity as the core bottleneck.
- **Star rating is a weaker, secondary factor.** Error rates vary by only ~2 percentage points across rating groups, with high-rated (4–5 star) reviews slightly harder to classify due to overlapping "positive, superlative-heavy" language between genuine and fake reviews.
- **The hardest and most consequential subgroup is identified.** Short reviews with high star ratings form the largest and most error-prone cell (Logistic Regression 19.9%, BERT 13.6%, n = 2,336) — and it's exactly where BERT's contextual encoding adds the most value (a 6.3 percentage point improvement).
- **Findings translate into a practical deployment recommendation.** A tiered moderation architecture is proposed: Logistic Regression as a fast, interpretable first-pass filter, with short/high-rated reviews escalated to BERT or human review.

## Repository Structure

```
.
├── NLP Fake Review Detection.ipynb   # Full analysis: data prep, EDA, modelling, error analysis
├── requirements.txt                # Python dependencies
├── README.md
└── outputs/                        # Generated on first run
    ├── figures/                    # EDA and error-analysis plots
    ├── tables/
    ├── metrics/                    # CSVs: predictions, per-subgroup error rates
    └── models/                     # Serialised Logistic Regression + BERT checkpoint
```

## Data

- **Source:** [`theArijitDas/Fake-Reviews-Dataset`](https://huggingface.co/datasets/theArijitDas/Fake-Reviews-Dataset) on the Hugging Face Hub, loaded directly via the `datasets` library — no manual download needed.
- **Scope:** Binary labels distinguish human-written (genuine) from computer-generated (fake) English-language Amazon product reviews. Findings should be read within this scope, not generalised to all forms of deceptive reviewing.

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

NLTK resources (`punkt_tab`, `wordnet`, `stopwords`) are downloaded automatically the first time the notebook runs.

## Running the Notebook

```bash
jupyter notebook Team8_MSIN0221_Notebook.ipynb
```

Run cells top to bottom — the "Technical Setup" section must execute first, as it defines shared configuration, output directories, and preprocessing utilities used throughout.

**Note:** Fine-tuning BERT is computationally intensive; a GPU is strongly recommended. The training loop (2 epochs, batch size 16) will run on CPU but considerably more slowly.

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF unigrams | 0.860 | 0.865 | 0.853 | 0.859 |
| TF-IDF bigrams | 0.879 | 0.887 | 0.868 | 0.877 |
| Metadata only | 0.525 | 0.518 | 0.707 | 0.598 |
| TF-IDF + metadata | 0.859 | 0.864 | 0.853 | 0.859 |
| **LR benchmark (balanced)** | 0.885 | 0.887 | 0.883 | 0.885 |
| **BERT (bert-base-uncased)** | **0.926** | 0.897 | **0.962** | **0.928** |

See the report and notebook for the full subgroup error-rate breakdown by review length and star rating.

## Limitations

The dataset contains only computer-generated fake reviews (not human-written deceptive ones), comes from a single platform/language, and BERT was fine-tuned for just two epochs with a 128-token limit. Full details are in Section 5 of the report.

## References

Key sources cited in the report include Devlin et al. (2019) on BERT, and Duma et al. (2024) on fake review detection literature — see the report's reference list for the complete bibliography.
# NLP-Fake-Review-Detection
