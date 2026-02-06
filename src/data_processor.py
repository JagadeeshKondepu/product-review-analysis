import pandas as pd
import re
from textblob import TextBlob
import nltk
from collections import Counter

class DataProcessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def extract_brand(self, text, name):
        if pd.notna(name):
            words = str(name).split()
            if words:
                return words[0]
        return "Unknown"
    
    def extract_category(self, text):
        categories = {
            'electronics': ['phone', 'laptop', 'tablet', 'computer', 'headphone', 'speaker', 'camera'],
            'clothing': ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'coat'],
            'home': ['furniture', 'bed', 'chair', 'table', 'lamp'],
            'beauty': ['cream', 'lotion', 'makeup', 'perfume', 'shampoo']
        }
        text_lower = str(text).lower()
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return cat
        return 'general'
    
    def analyze_sentiment(self, text):
        if pd.isna(text):
            return 0, 'neutral'
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return polarity, 'positive'
        elif polarity < -0.1:
            return polarity, 'negative'
        return polarity, 'neutral'
    
    def process_reviews(self, df):
        text_col = 'reviews.text' if 'reviews.text' in df.columns else 'Text'
        name_col = 'name' if 'name' in df.columns else 'ProductName'
        rating_col = 'reviews.rating' if 'reviews.rating' in df.columns else 'Rating'
        
        df['brand'] = df.apply(lambda x: self.extract_brand(x.get(text_col, ''), x.get(name_col, '')), axis=1)
        df['category'] = df[text_col].apply(self.extract_category) if text_col in df.columns else 'general'
        
        if text_col in df.columns:
            sentiments = df[text_col].apply(self.analyze_sentiment)
            df['sentiment_score'] = sentiments.apply(lambda x: x[0])
            df['sentiment_label'] = sentiments.apply(lambda x: x[1])
        
        if rating_col in df.columns:
            df['rating'] = pd.to_numeric(df[rating_col], errors='coerce')
        
        return df
