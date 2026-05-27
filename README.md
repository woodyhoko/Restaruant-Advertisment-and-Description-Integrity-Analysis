# Restaurant Advertisement & Description Integrity Analysis

An NLP research project that **audits the consistency between restaurant self-descriptions and customer reviews**, identifying misleading or inflated advertising claims.

📊 **[View Presentation](https://prezi.com/view/95Q588AKgodUxM5NISLN/)**

---

## Overview

Restaurant listings often contain marketing language that may not reflect the actual customer experience. This project uses text mining and natural language processing to:

1. Extract **sentiment and key claims** from restaurant self-descriptions (advertising copy)
2. Analyze **customer review corpora** for the same establishments
3. Compute an **integrity score** measuring alignment between advertised promises and reported experience

---

## Repository Contents

| File | Description |
|---|---|
| `project3.ipynb` | Main analysis notebook — data loading, NLP pipeline, integrity scoring |
| `comp4332_project2.py` | Supporting module for text feature extraction |
| `project4.py` | Extended analysis / follow-up experiments |
| `Group04_project3.zip` | Full project package including report |

---

## Methods

- **TF-IDF** vectorization of advertisement and review text
- **Sentiment analysis** (lexicon-based and model-based)
- **Cosine similarity** between advertisement claims and review topics
- Topic modeling for theme extraction

---

## Stack

- Python, NLTK, scikit-learn
- Pandas, NumPy
- Jupyter Notebooks

---

## Usage

```bash
pip install nltk scikit-learn pandas jupyter
jupyter notebook project3.ipynb
```

