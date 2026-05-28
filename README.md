# Restaurant Advertisement & Description Integrity Analysis

*Detecting misleading claims in restaurant listings using NLP consistency analysis between advertising copy and customer reviews.*

📊 **[View Presentation](https://prezi.com/view/95Q588AKgodUxM5NISLN/)**

---

## 1. Motivation

Restaurant discovery platforms (Yelp, Google Maps, TripAdvisor) allow operators to write free-form self-descriptions alongside algorithmically collected customer reviews. This creates a systematic **information asymmetry**: operators craft descriptions to maximize appeal, while reviews represent unfiltered real experiences. Prior work on deceptive opinion spam (Ott et al. 2011; Mukherjee et al. 2012) has focused on fake reviews. This project takes the complementary angle: auditing whether the *operator's own claims* are supported by the review corpus.

**Research question:** Given a restaurant's self-description and a set of customer reviews, can we automatically quantify the extent to which the description's claims are corroborated — or contradicted — by the reviews?

---

## 2. Problem formulation

Let *D* be the restaurant's advertisement text and *R* = {*r₁*, …, *rₙ*} be the set of customer reviews. We define an **integrity score** *S(D, R)* ∈ [0, 1] as a measure of semantic consistency:

```
S(D, R) = f(D, R)    where higher = better agreement between ad claims and reviews
```

The goal is to surface restaurants where *S* is anomalously low — potential cases of misleading advertising.

---

## 3. NLP pipeline

### 3.1 Text preprocessing

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return tokens
```

### 3.2 TF-IDF representation

Both the advertisement *D* and the concatenated review corpus *R̂* are vectorized with **TF-IDF** (Term Frequency–Inverse Document Frequency):

```
tfidf(t, d) = tf(t, d) × log(N / df(t))
```

The IDF is computed over the full corpus of all restaurants, so terms that appear in every listing (e.g. "restaurant", "food") receive low weight, while distinctive claims ("michelin", "authentic Sichuan", "24-hour") receive high weight.

### 3.3 Cosine similarity

The primary integrity signal is the **cosine similarity** between the TF-IDF vectors of *D* and *R̂*:

```python
S_cosine = cosine_similarity(tfidf(D), tfidf(R̂))
         = dot(v_D, v_R) / (||v_D|| · ||v_R||)
```

### 3.4 Sentiment alignment

A second signal computes the **sentiment gap** between description and reviews:

```python
# VADER lexicon-based sentiment (works well on short, informal text)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sent_D = sia.polarity_scores(D)['compound']        # description sentiment
sent_R = np.mean([sia.polarity_scores(r)['compound'] for r in R])  # avg review
sentiment_gap = sent_D - sent_R                    # positive = overclaiming
```

A large positive `sentiment_gap` (glowing description, tepid reviews) is a strong integrity signal.

### 3.5 Topic consistency (LDA)

**Latent Dirichlet Allocation** (Blei et al. 2003) extracts topic distributions from both documents. The **Jensen–Shannon divergence** between the topic distributions measures thematic consistency:

```
JSD(P_D || P_R) ∈ [0, 1]    (0 = identical topics, 1 = disjoint)
```

A restaurant that advertises "romantic atmosphere / fine dining" but whose reviews predominantly discuss "fast food / waiting time" will show high JSD.

### 3.6 Composite integrity score

```python
S = (1 - w₁) * S_cosine  +  w₂ * (1 - |sentiment_gap|)  +  w₃ * (1 - JSD)
```

Weights *w₁, w₂, w₃* are set empirically (default 0.4 / 0.3 / 0.3) and can be tuned on labelled data.

---

## 4. Key findings

- High-scoring (high integrity) restaurants tend to use specific, verifiable claims ("wood-fired", "gluten-free menu") rather than vague superlatives ("best", "amazing")
- Chains show systematically higher integrity scores than independent restaurants — standardized descriptions match standardized products
- Most impactful low-integrity signal: sentiment gap > 0.3 between description and reviews
- LDA topic mismatch is the weakest signal in isolation but provides complementary signal when combined with cosine similarity

---

## 5. Repository contents

| File | Description |
|---|---|
| `project3.ipynb` | Main analysis — data loading, NLP pipeline, scoring, visualization |
| `comp4332_project2.py` | Feature extraction module (TF-IDF, sentiment) |
| `project4.py` | Extended analysis / ablation experiments |
| `Group04_project3.zip` | Full project package with report |

---

## 6. Stack

Python · NLTK · VADER · scikit-learn · Gensim (LDA) · Pandas · Jupyter

---

## 7. Run

```bash
pip install nltk vaderSentiment scikit-learn gensim pandas jupyter
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
jupyter notebook project3.ipynb
```

---

## 8. References

1. M. Ott et al. "Finding Deceptive Opinion Spam by Any Stretch of the Imagination." *ACL '11*, 2011.
2. A. Mukherjee et al. "Spotting Fake Reviewer Groups in Consumer Reviews." *WWW '12*, 2012.
3. D. Blei, A. Ng, M. Jordan. "Latent Dirichlet Allocation." *JMLR*, 3:993–1022, 2003.
4. C. J. Hutto and E. Gilbert. "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *ICWSM '14*, 2014.
