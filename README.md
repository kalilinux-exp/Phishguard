# 🛡️ PhishGuard — Phishing Detector

> Browser-based phishing detection with three analysis layers — message patterns, domain verification, and email header inspection. No data ever leaves your device.

**[🌐 Live Demo](https://kalilinux-exp.github.io/Phishguard)** · Built by Kalixte Petrof

---

## What is PhishGuard?

Phishing scams cost Americans billions of dollars every year. Existing spam filters either send your private messages to company servers, or rely on simple word-matching that misses sophisticated attacks.

PhishGuard solves both problems — everything runs in your browser, and three analysis layers catch what a single check would miss.

---

## How It Works

### Layer 1 — Message Analysis
Scans the message body for urgency language, suspicious calls-to-action, prize/reward bait, requests for sensitive info, shortened URLs, and time pressure tactics. Also checks any URLs in the message against a verified domain list.

### Layer 2 — Sender & Domain Verification
Cross-references the sender's email domain against 30+ verified company domains. Catches:
- **Typosquatting** — `paypa1.com`, `micros0ft.com`, `wellsfarg0.com`
- **Free provider abuse** — real companies never send account alerts from Gmail or Yahoo
- **Impersonation** — domains that contain a company name but aren't the real domain
- **Suspicious TLDs** — `.xyz`, `.top`, `.tk`, `.ru`, etc.

### Layer 3 — Email Header Analysis
Paste raw email headers to get a full routing and authentication breakdown:
- **Received chain** — shows every server hop the email passed through, with timestamps and delays
- **SPF / DKIM / DMARC** — flags authentication failures that indicate a forged sender
- **Reply-To mismatch** — catches reply hijacking when Reply-To differs from From
- **Return-Path mismatch** — flags when bounce emails are redirected to a different domain
- **X-Spam-Status** — surfaces spam filter verdicts baked into the headers

---

## BERT Research Model

Alongside the browser tool, a standalone Python BERT classifier was trained on the UCI SMS Spam Collection dataset:

| Model | Accuracy | Phishing Recall |
|---|---|---|
| Naive Bayes (baseline) | 97.85% | ~75% |
| Logistic Regression | 96.41% | 75% |
| **BERT (fine-tuned)** | **97.4%+** | **~95%** |

Trained on the [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,572 messages).

**Key finding:** BERT's context understanding significantly improves recall on sophisticated phishing messages that bypass keyword-based filters — particularly messages that use legitimate-sounding language paired with suspicious sender domains.

> Note: The BERT model runs via the Python scripts. The live web demo uses heuristic analysis that runs entirely client-side with no server or dependencies.

---

## Project Structure

```
phishguard/
├── phishing_detector_site.html   # Live web demo — open in any browser, no server needed
├── enhanced_detector.py          # Full CLI detector: BERT + domain verification
├── phishing_detector.py          # Baseline classifiers (Naive Bayes + Logistic Regression)
├── phishing_detector_bert.py     # BERT fine-tuning script
├── save_model.py                 # Train and save BERT model to disk
└── check_message.py              # Load saved model and check messages instantly
```

---

## Running Locally

### Requirements
```bash
pip install pandas scikit-learn transformers torch
```

### Web Demo
Open `phishing_detector_site.html` in any browser. No install needed.

### Basic Model (fast)
```bash
# Download spam.csv from Kaggle first
python phishing_detector.py
```

### BERT Model
```bash
# Train and save (~30-45 mins on CPU, much faster with GPU)
python save_model.py

# Check messages instantly after saving
python check_message.py
```

### Full Detector (BERT + domain verification)
```bash
python enhanced_detector.py
```

---

## Example Output

**Web demo — message analysis:**
```
Message: "URGENT: Your bank account has been suspended. Verify: bit.ly/2xR9m"
Sender:  security@wellsfarg0.com
Result:  High Risk — Likely Phishing (100%)
         → Typosquatting detected: wellsfarg0.com is a fake version of wellsfargo.com
         → Uses urgency language
         → Contains a shortened URL
```

**Python BERT model:**
```
Message: "Hey, are we still on for dinner tonight at 7?"
Result:  REAL (0.0% chance of being fake)
```

---

## Key Research Findings

1. **Subdomain false positives** — Early versions incorrectly flagged legitimate corporate emails from subdomains like `alerts@notify.wellsfargo.com`. Fixed by implementing subdomain-aware domain matching.

2. **Contextual messages** — Formal-sounding legitimate messages (appointment reminders, shipping confirmations) can score borderline false positives. BERT handles these significantly better than keyword classifiers.

3. **Header analysis gaps** — Many phishing emails pass message-level checks but fail authentication entirely (SPF fail + DKIM fail + DMARC fail). Header analysis catches these cases that content scanning alone misses.

4. **The privacy gap** — No major consumer phishing detector currently runs entirely on-device. This remains an unsolved problem in mobile security.

---

## Technologies

- **Python** — core language
- **BERT** (`bert-base-uncased`) — Google's pre-trained language model via HuggingFace Transformers
- **PyTorch** — model training and inference
- **scikit-learn** — baseline classifiers and evaluation
- **HTML / CSS / JavaScript** — web demo (no framework, zero dependencies)

---

## About

Built as an independent research project exploring the application of transformer-based language models and header forensics to real-world cybersecurity problems.

**Kalixte Petrof** — High school researcher interested in cybersecurity, AI, and their intersection.

---

*⚠️ PhishGuard is a research project. No detector is 100% accurate — always verify suspicious messages through official channels.*
