# ============================================
# SMS Phishing Detector - BERT Upgrade
# Phase 2 (Advanced): Smarter AI that actually
# understands language, not just word counts
# ============================================

# STEP 0: Install these first (run in terminal):
# pip install transformers torch pandas scikit-learn

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ============================================
# QUICK CHECK: Does your computer have a GPU?
# If yes, BERT runs faster. If no, it uses CPU
# which is slower but still works fine.
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("No GPU found — running on CPU. This will take ~10-15 mins to train. That's okay!")
print()

# ============================================
# STEP 1: Load the dataset (same spam.csv)
# ============================================

print("Loading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(f"Dataset loaded! Total messages: {len(df)}")
print(f"Real messages (ham): {len(df[df['label'] == 0])}")
print(f"Fake/spam messages: {len(df[df['label'] == 1])}")
print()

# ============================================
# STEP 2: Split into train and test sets
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    df["message"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Training on {len(X_train)} messages...")
print(f"Testing on {len(X_test)} messages...")
print()

# ============================================
# STEP 3: Load BERT
# This downloads Google's pre-trained BERT model
# the first time (~400MB). After that it's cached.
# ============================================

print("Loading BERT model and tokenizer (may download on first run)...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # 2 labels: REAL or FAKE
)
model = model.to(device)
print("BERT loaded!")
print()

# ============================================
# STEP 4: Prepare data for BERT
# BERT needs text converted into tokens
# (numbers it understands) with a max length
# ============================================

class SMSDataset(Dataset):
    def __init__(self, messages, labels, tokenizer, max_len=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.messages[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SMSDataset(X_train, y_train, tokenizer)
test_dataset = SMSDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ============================================
# STEP 5: Fine-tune BERT on your phishing data
# This is where the magic happens —
# BERT already knows language, now it learns
# what phishing looks like specifically
# ============================================

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("Fine-tuning BERT on phishing data...")
print("(This is the slow part — grab a snack!)")
print()

EPOCHS = 3  # 3 passes through the data is enough

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_num, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches
        if (batch_num + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_num+1}/{len(train_loader)} | Loss: {total_loss/(batch_num+1):.4f}")

    train_acc = correct / total * 100
    print(f"Epoch {epoch+1} complete! Training Accuracy: {train_acc:.2f}%")
    print()

# ============================================
# STEP 6: Test BERT on unseen messages
# ============================================

print("Testing BERT on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nBERT Accuracy: {accuracy * 100:.2f}%")
print()
print("=== BERT Full Report ===")
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

# ============================================
# STEP 7: Test it on your own messages!
# Same as before but now powered by BERT
# ============================================

def check_text(message):
    """
    Takes any text message and tells you if
    it's REAL (ham) or FAKE (phishing/spam)
    using BERT's understanding of language.
    """
    model.eval()
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
        prediction = probs.argmax(dim=1).item()
        fake_prob = probs[0][1].item() * 100

    label = "FAKE" if prediction == 1 else "REAL"
    print(f"\nMessage: \"{message}\"")
    print(f"Result: {label} ({fake_prob:.1f}% chance of being fake)")
    return label

# --- Phishing messages ---
print("\n=== Testing on example messages ===")
print("\n🚨 These should all be FAKE:")
check_text("URGENT: Your bank account has been suspended. Click here to verify: bit.ly/2xR9m")
check_text("Congratulations! You've won a $1000 gift card. Claim now: freeprize.net/claim")
check_text("Your package could not be delivered. Update your address: track-pkg.com/update")
check_text("SunPass: You have an unpaid toll balance of $3.47. Pay now: sunpass-pay.com/balance")
check_text("Your Netflix subscription will auto-renew for $99.99 today. Cancel here: netflix-cancel.com/stop")

# --- Real messages ---
print("\n✅ These should all be REAL:")
check_text("Hey, are we still on for dinner tonight at 7?")
check_text("Mom, can you pick me up from practice at 5?")
check_text("Dude did you see that game last night?? Insane ending")
check_text("Can you grab milk on your way home? We're all out")
check_text("Reminder: your dentist appointment is tomorrow at 10am. Reply C to confirm")

# --- Try your own! ---
# check_text("Type any message here")
