import pandas as pd
import re

# Load data
df = pd.read_csv("../data/youtube_comments.csv")


# Clean comments
def clean_comment(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text


df["clean_comment"] = df["comment"].apply(clean_comment)

# Save cleaned data
df.to_csv("../data/youtube_comments_cleaned.csv", index=False)

print("✅ Comments cleaned and saved to data/youtube_comments_cleaned.csv")
