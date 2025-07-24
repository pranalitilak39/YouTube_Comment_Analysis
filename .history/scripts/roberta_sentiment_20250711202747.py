from transformers import pipeline
import pandas as pd

# Step 1: Load cleaned comments
df = pd.read_csv("../data/youtube_comments_cleaned.csv")

# Step 2: Load RoBERTa sentiment analysis pipeline
# The model is automatically downloaded first time you run this
classifier = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
)


# Step 3: Function to predict sentiment
def predict_sentiment(comment):
    result = classifier(comment)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score


# Step 4: Apply prediction to all comments

# Step 5: Add predictions to dataframe
df["sentiment"] = sentiments
df["score"] = scores

# Step 6: Save results to outputs folder
df.to_csv("../outputs/youtube_comments_with_sentiment.csv", index=False)

print(
    "✅ Sentiment analysis completed and saved to outputs/youtube_comments_with_sentiment.csv"
)
