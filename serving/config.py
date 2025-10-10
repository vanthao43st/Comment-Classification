import torch
import os
import re
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # thư mục 'serving'
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "phobert_sentiment"))

URL_REGEX     = re.compile(r'(https?://[^\s]+|www\.[^\s]+)')
EMAIL_REGEX   = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
MENTION_REGEX = re.compile(r'@[A-Za-z0-9_]+')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}

# Keyword spam
SPAM_KEYWORDS = {"camp", "vé", "lazada", "tiki", "feedback", "checklegit"}