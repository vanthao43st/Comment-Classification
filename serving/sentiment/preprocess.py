import re
from underthesea import word_tokenize
from config import URL_REGEX, EMAIL_REGEX, MENTION_REGEX

def basic_clean(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""

    # Replace URLs, emails, mentions
    text = URL_REGEX.sub(" <URL> ", text)
    text = EMAIL_REGEX.sub(" <EMAIL> ", text)
    text = MENTION_REGEX.sub(" @user ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_segment_vietnamese(text):
    """Vietnamese word segmentation for PhoBERT"""
    try:
        return word_tokenize(text, format="text").replace(" ", "_")
    except:
        return text.replace(" ", "_")