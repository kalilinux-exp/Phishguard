# ============================================
# SMS Phishing Detector - Save & Load Model
# Run this ONCE after training to save your
# model so you never have to retrain again!
# ============================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ============================================
# STEP 1: Check device
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# STEP 2: Load and train BERT again
# (same as before — we need to train it
# one more time so we can then save it)
# ============================================

print("\nLoading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = model.to(device)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("\nTraining BERT (3 epochs)...")
print("This is the last time you'll ever have to wait for this!\n")

for epoch in range(3):
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

        if (batch_num + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/3 | Batch {batch_num+1}/{len(train_loader)} | Loss: {total_loss/(batch_num+1):.4f}")

    print(f"Epoch {epoch+1} complete! Accuracy: {correct/total*100:.2f}%\n")

# ============================================
# STEP 3: SAVE the model
# This creates a folder called "saved_model"
# in your Downloads folder with everything
# needed to reload it instantly next time
# ============================================

print("Saving model to 'saved_model' folder...")
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
print("Model saved successfully!")
print("\nYou will NEVER have to train it again.")
print("Next time, just run: python check_message.py")
