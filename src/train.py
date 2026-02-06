import pandas as pd
import pickle
import sys
from data_processor import DataProcessor
from recommender import RecommendationSystem

def train_model(data_path, output_path='models/'):
    print("Loading data...")
    processor = DataProcessor()
    df = processor.load_data(data_path)
    
    print(f"Processing {len(df)} reviews...")
    df_processed = processor.process_reviews(df)
    
    print("Training recommendation system...")
    recommender = RecommendationSystem()
    recommender.train(df_processed)
    
    print("Saving models...")
    with open(f'{output_path}processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    with open(f'{output_path}recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    df_processed.to_csv(f'{output_path}processed_data.csv', index=False)
    
    print("\nTraining Summary:")
    print(f"Total reviews: {len(df_processed)}")
    print(f"Unique brands: {df_processed['brand'].nunique()}")
    print(f"Unique categories: {df_processed['category'].nunique()}")
    print(f"\nSentiment distribution:")
    print(df_processed['sentiment_label'].value_counts())
    print("\nModels saved successfully!")

if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/GrammarandProductReviews.csv'
    train_model(data_path)
