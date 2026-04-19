# ============================================
# check_message.py
# Load your saved BERT model and check any
# message instantly — no retraining needed!
# ============================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Load the saved model (takes ~10 seconds)
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading your saved model...")
tokenizer = BertTokenizer.from_pretrained("saved_model")
model = BertForSequenceClassification.from_pretrained("saved_model")
model = model.to(device)
model.eval()
print("Ready! Let's check some messages.\n")

# ============================================
# The check function
# ============================================

def check_text(message):
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
    print(f"Message: \"{message}\"")
    print(f"Result:  {label} ({fake_prob:.1f}% chance of being fake)\n")
    return label

# ============================================
# Test your messages here!
# Add or remove check_text() lines below
# ============================================

check_text("URGENT: Your bank account has been suspended. Click here: bit.ly/2xR9m")
check_text("Hey, are we still on for dinner tonight at 7?")
check_text("Congratulations! You've won a $1000 gift card. Claim now: freeprize.net/claim")
check_text("Mom, can you pick me up from practice at 5?")
check_text("Your package could not be delivered. Update your address: track-pkg.com/update")

# --- Add your own messages below! ---
# check_text("Type any message here")
