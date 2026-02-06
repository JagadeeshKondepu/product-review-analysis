import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.product_features = None
        self.df = None
    
    def train(self, df):
        self.df = df.copy()
        text_col = 'reviews.text' if 'reviews.text' in df.columns else 'Text'
        
        if text_col in df.columns:
            texts = df[text_col].fillna('')
            self.product_features = self.vectorizer.fit_transform(texts)
        
        return self
    
    def get_brand_affinity(self, user_reviews):
        brands = user_reviews['brand'].value_counts()
        sentiments = user_reviews.groupby('brand')['sentiment_score'].mean()
        affinity = pd.DataFrame({'count': brands, 'avg_sentiment': sentiments})
        affinity['score'] = affinity['count'] * (1 + affinity['avg_sentiment'])
        return affinity.sort_values('score', ascending=False)
    
    def get_category_affinity(self, user_reviews):
        categories = user_reviews['category'].value_counts()
        sentiments = user_reviews.groupby('category')['sentiment_score'].mean()
        affinity = pd.DataFrame({'count': categories, 'avg_sentiment': sentiments})
        affinity['score'] = affinity['count'] * (1 + affinity['avg_sentiment'])
        return affinity.sort_values('score', ascending=False)
    
    def recommend(self, user_history_indices, top_n=5, sentiment_threshold=0.0):
        if self.df is None:
            return []
        
        user_reviews = self.df.iloc[user_history_indices]
        brand_affinity = self.get_brand_affinity(user_reviews)
        category_affinity = self.get_category_affinity(user_reviews)
        
        preferred_brands = brand_affinity.head(3).index.tolist()
        preferred_categories = category_affinity.head(3).index.tolist()
        
        candidates = self.df[
            (self.df['brand'].isin(preferred_brands) | self.df['category'].isin(preferred_categories)) &
            (self.df['sentiment_score'] >= sentiment_threshold) &
            (~self.df.index.isin(user_history_indices))
        ].copy()
        
        if len(candidates) == 0:
            candidates = self.df[~self.df.index.isin(user_history_indices)].copy()
        
        if 'rating' in candidates.columns:
            candidates['score'] = candidates['sentiment_score'] * 0.5 + candidates['rating'] / 5.0 * 0.5
        else:
            candidates['score'] = candidates['sentiment_score']
        
        recommendations = candidates.nlargest(top_n, 'score')
        return recommendations.index.tolist()
