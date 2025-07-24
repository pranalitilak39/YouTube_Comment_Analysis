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


# Step 4: Apply prediction to all comments
# Step 4: Apply prediction to all comments safely
sentiments = []
scores = []

for comment in df["clean_comment"]:
    if pd.isnull(comment) or comment.strip() == "":
        sentiments.append("neutral")
        scores.append(0.0)
    else:
        try:
            label, score = predict_sentiment(str(comment))
            sentiments.append(label)
            scores.append(score)
        except Exception as e:
            print(f"Error processing comment: {comment}, Error: {e}")
            sentiments.append("error")
            scores.append(0.0)

# Step 5: Add predictions to dataframe
df["sentiment"] = sentiments
df["score"] = scores

# Step 6: Save results to outputs folder
df.to_csv("../outputs/youtube_comments_with_sentiment.csv", index=False)

print(
    "✅ Sentiment analysis completed and saved to outputs/youtube_comments_with_sentiment.csv"
)
