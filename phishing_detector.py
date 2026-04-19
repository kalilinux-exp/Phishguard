# ============================================
# SMS Phishing Detector - Starter Code
# Phase 2: Train your first classifier
# ============================================

# STEP 0: Install these first (run in terminal):
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# STEP 1: Load the dataset
# Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Save it as "spam.csv" in the same folder as this file
# ============================================

print("Loading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the two columns we need and rename them
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels: "ham" = 0 (safe), "spam" = 1 (phishing/spam)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(f"Dataset loaded! Total messages: {len(df)}")
print(f"Real messages (ham): {len(df[df['label'] == 0])}")
print(f"Fake/spam messages: {len(df[df['label'] == 1])}")
print()

# ============================================
# STEP 2: Split into training and test sets
# 80% of data trains the model, 20% tests it
# ============================================

X = df["message"]   # The text messages
y = df["label"]     # The labels (0 = safe, 1 = phishing)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} messages...")
print(f"Testing on {len(X_test)} messages...")
print()

# ============================================
# STEP 3: Convert text to numbers (TF-IDF)
# Computers can't read text — this turns words
# into numbers the model can understand
# ============================================

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ============================================
# STEP 4: Train two models and compare them
# ============================================

# --- Model 1: Naive Bayes (fast and simple) ---
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)
nb_accuracy = accuracy_score(y_test, nb_preds)
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")

# --- Model 2: Logistic Regression (smarter) ---
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)
lr_accuracy = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print()

# ============================================
# STEP 5: See a detailed report
# Precision = how often it's right when it says "phishing"
# Recall = how many real phishing messages it catches
# ============================================

print("=== Logistic Regression Full Report ===")
print(classification_report(y_test, lr_preds, target_names=["Real", "Fake"]))

# ============================================
# STEP 6: Test it on your OWN messages!
# ============================================

def check_text(message):
    """
    Takes a text message and tells you if it's REAL (ham) or FAKE (spam/phishing).
    """
    vec = vectorizer.transform([message])
    prediction = lr_model.predict(vec)[0]
    confidence = lr_model.predict_proba(vec)[0]

    label = "FAKE" if prediction == 1 else "REAL"
    score = confidence[1] * 100  # Fake probability as a percentage

    print(f"\nMessage: \"{message}\"")
    print(f"Result: {label} ({score:.1f}% chance of being fake)")
    return label

# --- Try these example messages ---
print("\n=== Testing on example messages ===")

# 🚨 FAKE messages
check_text("URGENT: Your bank account has been suspended. Click here to verify: bit.ly/2xR9m")
check_text("Congratulations! You've won a $1000 gift card. Claim now: freeprize.net/claim")
check_text("Your package could not be delivered. Update your address: track-pkg.com/update")
check_text("SunPass: You have an unpaid toll balance of $3.47. Pay now: sunpass-pay.com/balance")
check_text("Your Netflix subscription will auto-renew for $99.99 today. Cancel here: netflix-cancel.com/stop")

# ✅ REAL messages
check_text("Hey, are we still on for dinner tonight at 7?")
check_text("Mom, can you pick me up from practice at 5?")
check_text("Dude did you see that game last night?? Insane ending")
check_text("Can you grab milk on your way home? We're all out")
check_text("Reminder: your dentist appointment is tomorrow at 10am. Reply C to confirm")

# --- Try your own message here! ---
# check_text("Type any message here to test it")
