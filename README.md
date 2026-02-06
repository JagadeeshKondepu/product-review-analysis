# Product Review Sentiment Analysis & Recommendation System

A complete solution for analyzing customer reviews, extracting sentiment, and building a recommendation system refined by user preferences and sentiment scores.

## Features

- **Sentiment Analysis**: Analyzes review text to determine positive/negative/neutral sentiment
- **Brand & Category Extraction**: Extracts brands and product categories from reviews
- **Consumer Affinity Analysis**: Identifies user preferences for brands and categories
- **Sentiment-Refined Recommendations**: Recommends products based on user history and positive sentiment
- **Dockerized Deployment**: Easy training and inference with Docker containers

## Project Structure

```
.
├── src/
│   ├── data_processor.py    # Data processing and sentiment analysis
│   ├── recommender.py        # Recommendation system
│   ├── train.py              # Training pipeline
│   └── predict.py            # Inference pipeline
├── data/                     # Place your reviews.csv here
├── models/                   # Trained models saved here
├── Dockerfile.train          # Training container
├── Dockerfile.predict        # Prediction container
├── docker-compose.yml        # Orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+ OR Docker installed
- Dataset: https://lnkd.in/guXPbKge

### Option 1: Docker (Recommended)

**Setup:**
```bash
# Place GrammarandProductReviews.csv in data/ folder
mkdir data models
```

**Training:**
```bash
docker compose build train
docker compose run train
```

**Prediction:**
```bash
docker compose build predict
docker compose run predict
```

**Custom user history:**
```bash
docker compose run predict python src/predict.py "0,5,10"
```

### Option 2: Python Direct

**Install:**
```bash
pip install pandas numpy scikit-learn textblob nltk
```

**Training:**
```bash
python src/train.py data/GrammarandProductReviews.csv
```

**Prediction:**
```bash
python src/predict.py
python src/predict.py "0,5,10,15"
```

## How It Works

### 1. Data Processing
- Loads review data from CSV
- Extracts brand names from product names
- Categorizes products based on keywords
- Analyzes sentiment using TextBlob (polarity scores)
- Uses structured rating columns when available

### 2. Sentiment Analysis
- Polarity > 0.1: Positive
- Polarity < -0.1: Negative
- Otherwise: Neutral

### 3. Recommendation System
- Calculates brand affinity: frequency × (1 + avg_sentiment)
- Calculates category affinity: frequency × (1 + avg_sentiment)
- Filters candidates by preferred brands/categories
- Applies sentiment threshold (default: 0.0)
- Ranks by combined score: 50% sentiment + 50% rating

### 4. Training Pipeline
1. Load and process reviews
2. Extract features (brands, categories, sentiment)
3. Train TF-IDF vectorizer for text similarity
4. Save models and processed data

### 5. Inference Pipeline
1. Load trained models
2. Analyze user history
3. Identify brand/category preferences
4. Generate sentiment-refined recommendations

## Example Output

```
User History (3 items):
  - Samsung Galaxy S10 | Brand: Samsung | Sentiment: positive
  - Apple iPhone 11 | Brand: Apple | Sentiment: positive
  - Sony Headphones | Brand: Sony | Sentiment: neutral

Top 5 Recommendations:
1. Samsung Galaxy S20
   Brand: Samsung | Category: electronics
   Sentiment: positive (0.45)
   Rating: 4.5/5.0

2. Apple AirPods Pro
   Brand: Apple | Category: electronics
   Sentiment: positive (0.62)
   Rating: 4.7/5.0
...
```

## Validation

**Check training results:**
```bash
# View processed data
cat models/processed_data.csv | head

# Check model files exist
ls -la models/
```

**Test recommendations:**
```bash
# Test with different user histories
docker-compose run predict python src/predict.py "1,2,3"
docker-compose run predict python src/predict.py "10,20,30"
```

## Customization

**Adjust sentiment threshold:**
Edit `src/predict.py`:
```python
get_recommendations(user_indices, top_n=5, sentiment_threshold=0.2)
```

**Modify categories:**
Edit `src/data_processor.py` in the `extract_category` method.

**Change recommendation algorithm:**
Edit `src/recommender.py` in the `recommend` method.

## Technical Details

- **Sentiment Analysis**: TextBlob (rule-based, no training required)
- **Text Processing**: TF-IDF vectorization
- **Similarity**: Cosine similarity for content-based filtering
- **Scoring**: Weighted combination of sentiment and ratings

## Deployment Considerations

- Models are lightweight (< 10MB typically)
- Training time: ~1-5 minutes for 10K reviews
- Inference time: < 1 second per recommendation
- Scalable with batch processing
- Can be deployed to AWS ECS, Kubernetes, or serverless

## Future Enhancements

- Deep learning models (BERT, RoBERTa) for sentiment
- Collaborative filtering for recommendations
- Real-time streaming with Kafka
- A/B testing framework
- REST API with FastAPI
- Monitoring and logging

## License

MIT License
