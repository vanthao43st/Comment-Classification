import os, re, random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from underthesea import word_tokenize
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer, EarlyStoppingCallback,
    AutoConfig
)
import wandb

# ==========================
#  CONFIG
# ==========================
MODEL_NAME = "vinai/phobert-large"
DATA_PATH = "./training/datasets/Text_Emotion_2.csv"
MODEL_SAVE_PATH = "./serving/models/phobert_sentiment"

# Label mapping
LABEL_MAP = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
TARGET_LABELS = [0, 1, 2]

# Text cleaning
URL_REGEX     = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
EMAIL_REGEX   = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
MENTION_REGEX = re.compile(r'@[A-Za-z0-9_]+')

# Keyword spam
SPAM_KEYWORDS = {"camp", "vé", "lazada", "tiki", "feedback", "checklegit"}

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")
print(f"📊 Target labels: {TARGET_LABELS}")



# ==========================
#  LOAD DATA
# ==========================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df_clean = df.dropna(subset=['comment_text', 'labels']).copy()
    df_clean['labels'] = df_clean['labels'].astype(int)
    df_clean = df_clean[df_clean['labels'].isin(TARGET_LABELS)].reset_index(drop=True)
    return df_clean




# ==========================
#  CLEANING
# ==========================
def replace_url_with_token(match):
    url = match.group(0)
    try:
        domain = urlparse(url).netloc.lower()
    except:
        domain = url.lower()

    # YouTube
    if "youtube.com" in domain or "youtu.be" in domain:
        return " <YOUTUBE> "

    # Facebook (nếu chứa keyword spam trong link thì thành SPAMURL)
    elif "facebook.com" in domain or "fb.com" in domain:
        low_url = url.lower()
        if any(kw in low_url for kw in SPAM_KEYWORDS):
            return " <SPAMURL> "
        else:
            return " <FACEBOOK> "

    # Mặc định tất cả domain khác = SPAM
    else:
        return " <SPAMURL> "

def basic_clean(text):
    if not isinstance(text, str):
        return ""

    # Thay URL theo rule
    text = URL_REGEX.sub(replace_url_with_token, text)

    # Email, mention
    text = EMAIL_REGEX.sub(" <EMAIL> ", text)
    text = MENTION_REGEX.sub(" @user ", text)

    # Chuẩn hóa khoảng trắng
    return re.sub(r"\s+", " ", text).strip()




# ==========================
#  SPLIT DATA
# ==========================
def data_split(df_clean):
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.2,
        stratify=df_clean['labels'],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['labels'],
        random_state=RANDOM_STATE
    )

    train_dataset = Dataset.from_pandas(
        train_df[['comment_text_clean', 'labels']].rename(columns={'comment_text_clean': 'comment_text'}),
        preserve_index=False
    )

    val_dataset = Dataset.from_pandas(
        val_df[['comment_text_clean', 'labels']].rename(columns={'comment_text_clean': 'comment_text'}),
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df[['comment_text_clean', 'labels']].rename(columns={'comment_text_clean': 'comment_text'}),
        preserve_index=False
    )

    return train_dataset, val_dataset, test_dataset




# ==========================
#  TOKENIZATION
# ==========================
def word_segment_vietnamese(text):
    """Vietnamese word segmentation for PhoBERT"""
    try:
        return word_tokenize(text, format="text").replace(" ", "_")
    except:
        return text.replace(" ", "_")

def tokenize_batch(tokenizer, batch):
    """Tokenize batch of texts with Vietnamese word segmentation"""
    segmented_texts = [word_segment_vietnamese(text) for text in batch['comment_text']]

    return tokenizer(
        segmented_texts,
        truncation=True,
        max_length=256,
        padding=False  # Dynamic padding will be handled by Trainer
    )





# ==========================
#  METRICS
# ==========================
def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }




# ==========================
#  MAIN
# ==========================
def main():
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15
    )

    print("🚀 Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    ).to(DEVICE)

    print("📚 Loading dataset...")
    df_clean = load_data()
    train_dataset, val_dataset, test_dataset = data_split(df_clean)

    print("🔄 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        do_train=True,
        do_eval=True,

        learning_rate=8e-6,              # quay về LR ổn định
        num_train_epochs=20,             # đủ lâu để hội tụ
        warmup_ratio=0.2,
        weight_decay=0.2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,

        # theo dõi & chọn theo macro-F1
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,             # THÊM MỚI
        save_steps=100,             # THÊM MỚI
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,

        # scheduler mượt hơn
        lr_scheduler_type="cosine",

        # ổn định/hiệu năng
        fp16=True,
        max_grad_norm=0.5,
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
        dataloader_num_workers=2,
        label_smoothing_factor=0.1,
        logging_dir="./training_logs"
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=5e-4
    )

    wandb.init(project="phobert_sentiment", config={
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16
    })

    print("⚙️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    # ⬇️ wandb sẽ tự động ghi log từ Trainer của HuggingFace
    trainer.train()

    # ✅ Kết thúc logging
    wandb.finish()

    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()