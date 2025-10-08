import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH, DEVICE, LABEL_MAP
from .preprocess import basic_clean, word_segment_vietnamese

class SentimentAnalyzer:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        cleaned_text = basic_clean(text)
        segmented_text = word_segment_vietnamese(cleaned_text)

        inputs = self.tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()

        results = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "segmented_text": segmented_text,
            "predicted_label": predicted_class,
            "predicted_class": LABEL_MAP[predicted_class],
            "probabilities": {
                LABEL_MAP[i]: prob.item()
                for i, prob in enumerate(probabilities.squeeze())
            }
        }

        return results