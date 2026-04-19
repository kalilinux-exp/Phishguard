# 🛡️ PhishGuard — AI-Powered Phishing Detector

> Detects phishing attempts in SMS messages and emails using BERT AI, domain verification, and typosquat detection — all running locally with zero data sent online.

**[🌐 Live Demo](https://kalilinux-exp.github.io/Phishguard)** · Built by Kalixte Petrof

---

## What is PhishGuard?

Phishing scams cost Americans billions of dollars every year. Existing spam filters either send your private messages to company servers, or rely on simple word-matching that misses sophisticated attacks.

PhishGuard is a research project that solves both problems:

- **Smarter detection** using Google's BERT language model, which understands context — not just keywords
- **Privacy first** — all analysis runs on your device. Your messages never leave your browser or computer
- **Three detection layers** working together for higher accuracy

---

## How It Works

### Layer 1 — BERT Message Analysis
A fine-tuned BERT model reads the full message and understands the meaning behind it. Unlike basic spam filters that count suspicious words, BERT understands that *"your account is suspended"* from an unknown sender is very different from *"hey are you coming to practice?"*

### Layer 2 — Domain Verification
Cross-references the sender's email domain against a verified list of 30+ major companies. Recognizes legitimate subdomains (like `notify.wellsfargo.com`) while flagging impersonation attempts.

### Layer 3 — Typosquat Detection
Catches common phishing tricks like replacing letters with numbers (`paypa1.com`, `micros0ft.com`, `wellsfarg0.com`) that sometimes trick us.

---

## Results

| Model | Accuracy | Phishing Recall |
|---|---|---|
| Naive Bayes (baseline) | 97.85% | ~75% |
| Logistic Regression | 96.41% | 75% |
| **BERT (fine-tuned)** | **97.4%+** | **~95%** |

Trained and tested on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,572 messages).

**Key finding:** BERT's context understanding significantly improves recall on sophisticated phishing messages that bypass keyword-based filters — particularly messages that use legitimate-sounding language with suspicious sender domains.

---

## Project Structure

```
phishguard/
├── phishing_detector_site.html   # Live web demo (works offline)
├── phishing_detector.py          # Basic classifier (Naive Bayes + Logistic Regression)
├── phishing_detector_bert.py     # BERT fine-tuning script
├── save_model.py                 # Save trained model to disk
├── check_message.py              # Load saved model and check messages
└── enhanced_detector.py          # Full detector with domain verification
```

---

## Running Locally

### Requirements
```bash
pip install pandas scikit-learn transformers torch
```

### Quick Start (Basic Model)
```bash
# Download spam.csv from Kaggle first
python phishing_detector.py
```

### BERT Model
```bash
# Train and save (takes ~30-45 mins on CPU)
python save_model.py

# Check messages instantly after saving
python check_message.py
```

### Web Demo
Just open `phishing_detector_site.html` in any browser. No server needed.

---

## Example Output

```
Message: "URGENT: Your bank account has been suspended. Verify: bit.ly/2xR9m"
Result:  FAKE (99.1% chance of being fake)

Message: "Hey, are we still on for dinner tonight at 7?"
Result:  REAL (0.0% chance of being fake)
```

---

## Key Research Findings

1. **Subdomain false positives** — Early versions incorrectly flagged legitimate corporate emails from subdomains like `alerts@notify.wellsfargo.com`. Fixed by implementing subdomain-aware domain matching.

2. **Contextual messages** — Formal-sounding legitimate messages (appointment reminders, shipping confirmations) can score borderline false positives. BERT handles these significantly better than keyword classifiers.

3. **The privacy gap** — No major consumer phishing detector currently runs entirely on-device for SMS. This remains an unsolved problem in mobile security.

---

## Technologies Used

- **Python** — core language
- **BERT** (`bert-base-uncased`) — Google's pre-trained language model via HuggingFace Transformers
- **PyTorch** — model training and inference
- **scikit-learn** — baseline classifiers and evaluation
- **HTML/CSS/JavaScript** — web demo (no framework, zero dependencies)

---

## About

Built as an independent research project exploring the application of transformer-based language models to real-world cybersecurity problems.

**Kalixte Petrof** ---- High school researcher interested in cybersecurity, AI, and their intersection.
(thats me!)
---

*⚠️ PhishGuard is a research project. No detector is 100% accurate,, — always verify suspicious messages through official channels.*
