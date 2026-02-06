import pickle
import sys
import pandas as pd

def load_models(model_path='models/'):
    with open(f'{model_path}recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)
    df = pd.read_csv(f'{model_path}processed_data.csv')
    return recommender, df

def get_recommendations(user_indices, top_n=5, sentiment_threshold=0.0):
    recommender, df = load_models()
    
    print(f"\nUser History ({len(user_indices)} items):")
    for idx in user_indices:
        row = df.iloc[idx]
        name_col = 'name' if 'name' in df.columns else 'ProductName'
        print(f"  - {row.get(name_col, 'Unknown')} | Brand: {row['brand']} | Sentiment: {row['sentiment_label']}")
    
    recommendations = recommender.recommend(user_indices, top_n, sentiment_threshold)
    
    print(f"\nTop {top_n} Recommendations:")
    for i, idx in enumerate(recommendations, 1):
        row = df.iloc[idx]
        name_col = 'name' if 'name' in df.columns else 'ProductName'
        print(f"{i}. {row.get(name_col, 'Unknown')}")
        print(f"   Brand: {row['brand']} | Category: {row['category']}")
        print(f"   Sentiment: {row['sentiment_label']} ({row['sentiment_score']:.2f})")
        if 'rating' in df.columns:
            print(f"   Rating: {row['rating']:.1f}/5.0")
    
    return recommendations

if __name__ == '__main__':
    user_indices = [0, 1, 2] if len(sys.argv) == 1 else list(map(int, sys.argv[1].split(',')))
    get_recommendations(user_indices)
