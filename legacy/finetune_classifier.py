"""
Path A — fine-tune an encoder (InLegalBERT) into a 3-class classifier.

Task modeled here: ONE axis — "is this memory a crime?" -> yes / no / unclear.
For your other axis (ethical good/bad) you train a SECOND, separate model with
the same script, just swapping the dataset. Don't share a head between axes.

Assumptions (change these if they don't hold):
  - Your labeled data is text + an integer label (0/1/2). A CSV with columns
    `text,label` is the easiest format; a tiny inline sample is included so this
    runs as-is.
  - 3 classes, possibly imbalanced (the "unclear" class is usually the smallest),
    so we optimize and report MACRO-F1, not raw accuracy.
  - Runs on your 4 GB GPU or CPU. InLegalBERT is ~110M params; no AirLLM needed.

Install:
  pip install "transformers>=4.46" datasets evaluate torch scikit-learn
"""

import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# --- Config -----------------------------------------------------------------
MODEL_NAME = "law-ai/InLegalBERT"   # swap to "microsoft/deberta-v3-base" for a
                                    # general (non-Indian-law) base, or a
                                    # multilingual encoder for EU/Danish text.
MAX_LEN = 256                       # raise if your memories are long.

id2label = {0: "no", 1: "yes", 2: "unclear"}
label2id = {v: k for k, v in id2label.items()}

# --- 1. Data ----------------------------------------------------------------
# Replace this toy sample with your real data, e.g.:
#   import pandas as pd
#   df = pd.read_csv("memories.csv")            # columns: text, label(0/1/2)
#   data = Dataset.from_pandas(df)
sample = {
    "text": [
        "He took the neighbour's bike without asking and sold it.",
        "She paid for her groceries and walked home.",
        "They argued loudly in the street late at night.",
        "I forged my manager's signature on the expense form.",
        "We donated our old clothes to the shelter.",
        "Someone was in the room but it's not clear what they did.",
    ],
    "label": [1, 0, 2, 1, 0, 2],  # yes / no / unclear
}
data = Dataset.from_dict(sample)

# Hold out a test split so the metrics mean something.
split = data.train_test_split(test_size=0.33, seed=42, stratify_by_column="label")
train_ds, eval_ds = split["train"], split["test"]

# --- 2. Tokenize ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 3. Model ---------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# --- 4. Metrics (macro-F1 is the one to watch with 3 imbalanced classes) ----
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# --- 5. Train ---------------------------------------------------------------
args = TrainingArguments(
    output_dir="./crime_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    eval_strategy="epoch",        # older transformers: evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print("Final eval:", trainer.evaluate())   # success check: macro_f1 on held-out set

# --- 6. Save ----------------------------------------------------------------
trainer.save_model("./crime_classifier")
tokenizer.save_pretrained("./crime_classifier")

# --- 7. Use it on a new memory ----------------------------------------------
import torch

def classify(text: str) -> str:
    inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return id2label[int(logits.argmax(-1))]

print(classify("He broke into the shop and emptied the till."))
