# ============================================
# SMS/Email Phishing Detector - Enhanced
# Now checks BOTH the message AND the sender
# email address for a combined verdict
# ============================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
import re
warnings.filterwarnings("ignore")

# ============================================
# KNOWN LEGITIMATE EMAIL DOMAINS
# These are verified official domains for
# major companies. Emails claiming to be from
# these companies should ONLY come from here.
# ============================================

VERIFIED_DOMAINS = {
    # Banks & Finance
    "chase.com":              "Chase Bank",
    "wellsfargo.com":         "Wells Fargo",
    "bankofamerica.com":      "Bank of America",
    "citibank.com":           "Citibank",
    "capitalone.com":         "Capital One",
    "discover.com":           "Discover",
    "americanexpress.com":    "American Express",
    "paypal.com":             "PayPal",
    "venmo.com":              "Venmo",
    "zelle.com":              "Zelle",

    # Streaming & Tech
    "netflix.com":            "Netflix",
    "spotify.com":            "Spotify",
    "apple.com":              "Apple",
    "google.com":             "Google",
    "microsoft.com":          "Microsoft",
    "amazon.com":             "Amazon",
    "amazonses.com":          "Amazon (email service)",

    # Shipping
    "usps.com":               "USPS",
    "ups.com":                "UPS",
    "fedex.com":              "FedEx",

    # Government
    "irs.gov":                "IRS",
    "ssa.gov":                "Social Security Administration",

    # Telecom
    "t-mobile.com":           "T-Mobile",
    "verizon.com":            "Verizon",
    "att.com":                "AT&T",

    # Social
    "instagram.com":          "Instagram",
    "facebook.com":           "Facebook",
    "tiktok.com":             "TikTok",
    "twitter.com":            "Twitter/X",
    "x.com":                  "Twitter/X",
    "linkedin.com":           "LinkedIn",
}

# ============================================
# SUSPICIOUS PATTERNS IN EMAIL ADDRESSES
# These are tricks phishers use to fake
# looking like a real company
# ============================================

SUSPICIOUS_PATTERNS = [
    r'\d{4,}',              # Lots of numbers: wellsfargo2847@...
    r'[_\-\.]{2,}',        # Double dashes/dots: well..fargo@...
    r'secure[\-_]',        # "secure-" prefix: secure-paypal@...
    r'[\-_]secure',        # "-secure" suffix
    r'alert[\-_]',         # "alert-" prefix
    r'support[\-_]',       # "support-" prefix  
    r'noreply[\-_\d]',     # Weird noreply variants
    r'verify[\-_]',        # "verify-" prefix
    r'update[\-_]',        # "update-" prefix
]

# ============================================
# COMMON TYPOSQUATTING TRICKS
# Phishers replace letters to look real:
# paypa1.com, arnazon.com, micros0ft.com
# ============================================

TYPOSQUAT_CHARS = {
    '0': 'o',   # micros0ft
    '1': 'l',   # paypa1
    '3': 'e',   # n3tflix
    '@': 'a',   # p@ypal (in domain)
    'rn': 'm',  # arnazon looks like amazon
    'vv': 'w',  # vvellsfargo
}

def extract_domain(email):
    """Pull the domain out of an email address."""
    email = email.strip().lower()
    if "@" in email:
        return email.split("@")[-1]
    return email

def check_sender(email):
    """
    Analyzes the sender email address and returns
    a verdict on whether it looks legitimate.
    """
    if not email or email.strip() == "":
        return None, "no_email"

    email = email.strip().lower()
    domain = extract_domain(email)
    results = []
    risk_score = 0

    # --- Check 1: Is it a known legitimate domain? ---
    if domain in VERIFIED_DOMAINS:
        company = VERIFIED_DOMAINS[domain]
        results.append(f"✅ Sender domain matches verified {company} domain")
        risk_score -= 30  # Good sign, lower risk
    else:
        # Check if it's PRETENDING to be a known company
        for verified_domain, company in VERIFIED_DOMAINS.items():
            company_name = verified_domain.split(".")[0]  # e.g. "wellsfargo"
            if company_name in domain and domain != verified_domain:
                results.append(f"🚨 Domain contains '{company_name}' but is NOT the official {company} domain!")
                results.append(f"   Official domain is: {verified_domain}")
                risk_score += 60  # Big red flag
                break
        else:
            results.append(f"⚠️  Sender domain '{domain}' is not in our verified list")
            risk_score += 15

    # --- Check 2: Suspicious patterns in email ---
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, email):
            results.append(f"🚨 Suspicious pattern detected in email address")
            risk_score += 25
            break

    # --- Check 3: Typosquatting detection ---
    for fake_char, real_char in TYPOSQUAT_CHARS.items():
        if fake_char in domain:
            modified = domain.replace(fake_char, real_char)
            if modified in VERIFIED_DOMAINS:
                company = VERIFIED_DOMAINS[modified]
                results.append(f"🚨 Domain looks like a fake version of {company}!")
                results.append(f"   '{domain}' vs real domain '{modified}'")
                risk_score += 80
                break

    # --- Check 4: Free email providers sending "official" alerts ---
    free_providers = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
    if domain in free_providers:
        results.append(f"⚠️  Sent from a free email provider ({domain})")
        results.append(f"   Real companies never send alerts from Gmail/Yahoo/Hotmail")
        risk_score += 40

    # --- Check 5: Weird TLDs ---
    suspicious_tlds = [".xyz", ".top", ".click", ".loan", ".gq", ".tk", ".ml", ".cf"]
    for tld in suspicious_tlds:
        if domain.endswith(tld):
            results.append(f"🚨 Suspicious domain extension: '{tld}'")
            risk_score += 35
            break

    return results, risk_score

# ============================================
# LOAD YOUR SAVED BERT MODEL
# Make sure you've run save_model.py first!
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading your saved BERT model...")
try:
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    model = BertForSequenceClassification.from_pretrained("saved_model")
    model = model.to(device)
    model.eval()
    print("Model loaded!\n")
    bert_available = True
except:
    print("Note: No saved model found. Run save_model.py first for AI analysis.")
    print("Running with email analysis only for now.\n")
    bert_available = False

# ============================================
# MAIN CHECK FUNCTION
# Combines BERT message analysis + sender check
# ============================================

def full_check(message, sender_email=""):
    print("=" * 60)
    print("PHISHING DETECTOR ANALYSIS")
    print("=" * 60)
    print(f"Message: \"{message}\"")
    if sender_email:
        print(f"Sender:  {sender_email}")
    print("-" * 60)

    total_risk = 0

    # --- BERT message analysis ---
    if bert_available:
        encoding = tokenizer(
            message,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            fake_prob = probs[0][1].item() * 100

        print(f"\n📱 MESSAGE ANALYSIS (AI):")
        print(f"   Phishing probability: {fake_prob:.1f}%")
        if fake_prob > 70:
            print(f"   Verdict: Message looks SUSPICIOUS")
            total_risk += 50
        elif fake_prob > 40:
            print(f"   Verdict: Message is BORDERLINE — check sender carefully")
            total_risk += 25
        else:
            print(f"   Verdict: Message content looks normal")

    # --- Sender email analysis ---
    if sender_email:
        print(f"\n📧 SENDER ANALYSIS:")
        sender_results, sender_risk = check_sender(sender_email)
        total_risk += max(0, sender_risk)
        for result in sender_results:
            print(f"   {result}")

    # --- Final combined verdict ---
    print(f"\n{'=' * 60}")
    if total_risk >= 60:
        print(f"🚨 FINAL VERDICT: HIGH RISK — This is likely PHISHING")
        print(f"   Do NOT click any links. Do NOT reply.")
        print(f"   Contact the company directly using their official website.")
    elif total_risk >= 30:
        print(f"⚠️  FINAL VERDICT: SUSPICIOUS — Proceed with caution")
        print(f"   Verify by contacting the company through official channels.")
    else:
        print(f"✅ FINAL VERDICT: Looks LEGITIMATE")
        print(f"   Still be cautious — no detector is perfect!")
    print("=" * 60)
    print()

# ============================================
# TEST IT OUT — try your own emails below!
# ============================================

# Real looking but fake bank email
full_check(
    message="Your account has been suspended. Verify your identity immediately to restore access.",
    sender_email="security@wellsfarg0.com"
)

# Actual legit Netflix email
full_check(
    message="Your Netflix subscription will renew on May 1st for $15.99.",
    sender_email="info@netflix.com"
)

# Gmail pretending to be PayPal
full_check(
    message="Your PayPal account has been limited. Click here to restore access.",
    sender_email="paypal.support@gmail.com"
)

# Real Chase email
full_check(
    message="Your statement is ready to view online.",
    sender_email="no-reply@chase.com"
)

# Fake IRS email
full_check(
    message="IRS NOTICE: Your tax refund of $892 is pending. Confirm your details now.",
    sender_email="refunds@irs-gov-refund.xyz"
)

# --- Try your own! ---
# full_check(
#     message="paste the email message here",
#     sender_email="sender@domain.com"
# )
