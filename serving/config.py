import torch
import re

MODEL_PATH = "./models/phobert_sentiment"  # Path to the fine-tuned model directory
URL_REGEX     = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
EMAIL_REGEX   = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
MENTION_REGEX = re.compile(r'@[A-Za-z0-9_]+')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}

# Keyword spam
SPAM_KEYWORDS = {"camp", "vé", "lazada", "tiki", "feedback", "checklegit"}