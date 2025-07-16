import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from googleapiclient.discovery import build


# Initialize YouTube API client
def youtube_scraper(video_id, api_key, max_comments=20):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_comments, 100),  # API max per page is 100
        textFormat="plainText",
    )
    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments


# 🌟 Cache model loading to prevent reloading each time
@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    return classifier


classifier = load_model()

# 🌟 Label mapping for human-readable output
label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


# 🌟 Function to predict sentiment
def predict_sentiment(comment):
    result = classifier(str(comment))
    label_raw = result[0]["label"]
    label = label_map.get(label_raw, "unknown")
    score = result[0]["score"]
    return label, score


# 🌟 Streamlit App UI
st.title("YouTube Comment Sentiment Analysis using RoBERTa")
st.write(
    "🔬 **Research-based live sentiment analysis with Transformer NLP model (RoBERTa).**"
)
import streamlit as st

st.title("YouTube Live Comment Scraper + Sentiment Analysis")

api_key = "YOUR_API_KEY"  # Replace with your actual API key

video_url = st.text_input("Enter YouTube Video URL:")
if video_url:
    if "watch?v=" in video_url:
        video_id = video_url.split("watch?v=")[-1][:11]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1][:11]
    else:
        st.error("Invalid YouTube URL")
        video_id = None

    if video_id:
        if st.button("Fetch Comments"):
            comments = youtube_scraper(video_id, api_key)
            st.write("### Sample Comments:")
            for c in comments:
                st.write("-", c)

            # Predict sentiment using your existing predict_sentiment function
            st.write("### Sentiment Analysis Results:")
            for c in comments:
                label, score = predict_sentiment(c)
                st.write(f"{c[:50]}... ➡ **{label}** ({score:.2f})")


# 🌟 Upload cleaned YouTube comments CSV
uploaded_file = st.file_uploader("Upload Cleaned YouTube Comments CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Comments Loaded")
    st.write(df.head())

    # 🌟 Live Sentiment Prediction Button
    if st.button("Run Live Sentiment Analysis"):
        sentiments = []
        scores = []

        # 🌟 Process each comment
        for comment in df["clean_comment"]:
            if pd.isnull(comment) or comment.strip() == "":
                sentiments.append("neutral")
                scores.append(0.0)
            else:
                label, score = predict_sentiment(comment)
                sentiments.append(label)
                scores.append(score)

        # 🌟 Add predictions to dataframe
        df["sentiment"] = sentiments
        df["score"] = scores

        # 🌟 Display final table
        st.write("### Comments with Sentiment")
        st.write(df.head(10))

        # 🌟 Sentiment Distribution Pie Chart
        st.write("### Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)

        # 🌟 Word Cloud Visualization
        st.write("### Word Cloud of Comments")
        text_combined = " ".join(df["clean_comment"].dropna().tolist())

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text_combined
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # 🌟 Download final processed CSV
        st.download_button(
            "Download Results CSV",
            data=df.to_csv(index=False),
            file_name="youtube_comments_with_sentiment.csv",
            mime="text/csv",
        )
