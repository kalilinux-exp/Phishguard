# PhishGuard: Context-Aware Phishing Detection Using On-Device BERT

**Kalixte Petrof** | Independent Research | 2025–2026
**GitHub:** github.com/kalilinux-exp/Phishguard | **Demo:** kalilinux-exp.github.io/Phishguard

---

## Problem

Phishing attacks cost Americans over $52 billion annually and remain one of the
most effective vectors for identity theft, financial fraud, and credential
compromise. Despite widespread awareness, phishing continues to succeed because
it exploits human trust rather than technical vulnerabilities — making it
difficult to defend against through traditional security measures alone.

Existing consumer-facing phishing detectors share two critical weaknesses: they
rely on keyword-matching that sophisticated attackers easily circumvent, and they
transmit message content to remote servers for analysis — creating a privacy
tradeoff that prevents adoption in sensitive contexts.

---

## Approach

This project investigates whether a fine-tuned transformer-based language model
can detect phishing messages more accurately than traditional classifiers while
running entirely on-device, eliminating the privacy concern entirely.

The system, PhishGuard, combines three detection layers:

1. **AI Message Analysis** — Google's BERT (bert-base-uncased) fine-tuned on
   labeled SMS data to classify messages based on contextual meaning rather than
   keyword frequency

2. **Domain Verification** — Real-time cross-referencing of sender email domains
   against a curated list of verified corporate domains, with subdomain-aware
   matching to reduce false positives

3. **Typosquat Detection** — Character substitution analysis to identify
   impersonation attempts (e.g. paypa1.com, wellsfarg0.com)

---

## Methods

The model was trained on the UCI SMS Spam Collection Dataset (5,572 labeled
messages) using an 80/20 train-test split. Three classifiers were evaluated:
Multinomial Naive Bayes, Logistic Regression with TF-IDF vectorization, and
BERT fine-tuned over three epochs using the AdamW optimizer (lr=2e-5, batch
size=16). All training was performed on CPU hardware without GPU acceleration.

---

## Results

| Model | Accuracy | Phishing Recall |
|---|---|---|
| Naive Bayes | 97.85% | ~75% |
| Logistic Regression | 96.41% | 75% |
| BERT (fine-tuned) | 97.4%+ | ~95% |

BERT demonstrated significantly higher recall on phishing messages — meaning
fewer dangerous messages slipped through undetected — while maintaining
comparable overall accuracy. The improvement was most pronounced on
contextually ambiguous messages that keyword-based models consistently
misclassified.

---

## Key Findings

**Finding 1 — Context matters more than keywords.**
Keyword-based classifiers correctly identified obvious phishing ("Congratulations,
you've won!") but failed on sophisticated messages using formal language with
suspicious sender domains. BERT's contextual understanding captured these cases
with significantly higher confidence.

**Finding 2 — Subdomain handling is a critical false positive source.**
Early testing revealed that legitimate corporate emails sent from subdomains
(e.g. alerts@notify.wellsfargo.com) were incorrectly flagged as phishing by
naive domain matching. A subdomain-aware verification system was implemented
to resolve this, reducing false positives on legitimate transactional emails.

**Finding 3 — The privacy gap remains unsolved at scale.**
No major consumer SMS phishing detector currently performs all analysis
on-device. PhishGuard demonstrates that a lightweight fine-tuned BERT model
can achieve near-commercial accuracy without transmitting message content —
suggesting that on-device deployment is technically feasible and practically
meaningful.

---

## Limitations & Future Work

The current model was trained exclusively on the UCI SMS dataset, which may not
fully represent modern phishing language patterns. Future work includes training
on larger, more recent datasets (including voice phishing transcripts), compressing
the model using TensorFlow Lite for true mobile deployment, and expanding domain
verification coverage. A longitudinal study measuring real-world false positive
and false negative rates against live phishing campaigns would strengthen the
findings considerably.

---

## Conclusion

PhishGuard demonstrates that transformer-based language models can meaningfully
outperform traditional keyword classifiers for phishing detection while preserving
user privacy through on-device inference. The subdomain false positive finding
suggests that domain verification systems require more nuanced implementation
than simple exact-match lookups — a detail absent from most existing approaches.
The complete codebase, trained model artifacts, and live demo are publicly
available at the links above.

---

*Independent research project. All training, testing, and development conducted
independently. Dataset: UCI SMS Spam Collection (Kaggle). Model: bert-base-uncased
via HuggingFace Transformers.*
