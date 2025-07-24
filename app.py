import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
import re
import time

# 🌟 Load YouTube API key securely from secrets
api_key = st.secrets["api"]["youtube_api_key"]


# 🌟 Initialize YouTube API client
def youtube_scraper(video_id, api_key, max_comments=20):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_comments, 100),
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


# 🌟 Clean text function for CSV uploads
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 🌟 Streamlit App UI
st.title("🎬 YouTube Comment Analysis using RoBERTa")

tab1, tab2 = st.tabs(["🔴 Live Scraping Analysis", "📁 File Upload Analysis"])

# ========== 🔴 Tab 1: Live Scraping ==========
with tab1:
    st.header("🔴 Live YouTube Comment Scraper + Sentiment Analysis")
    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        if "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[-1][:11]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1][:11]
        else:
            st.error("Invalid YouTube URL")
            video_id = None

        if video_id and st.button("Fetch Comments"):
            comments = youtube_scraper(video_id, api_key)
            st.write("### Sample Comments:")
            for c in comments:
                st.write("-", c)

            st.write("### Sentiment Analysis Results:")
            sentiments, scores = [], []
            for c in comments:
                label, score = predict_sentiment(c)
                sentiments.append(label)
                scores.append(score)
                st.write(f"{c[:50]}... ➡ **{label}** ({score:.2f})")

            st.write("### Sentiment Distribution")
            sentiment_counts = pd.Series(sentiments).value_counts()
            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.write("### Word Cloud of Comments")
            text_combined = " ".join(comments)
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(text_combined)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

# ========== 📁 Tab 2: File Upload ==========
with tab2:
    st.header("📁 Bulk Sentiment Analysis via CSV Upload")
    uploaded_file = st.file_uploader("Upload YouTube Comments CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        if "clean_comment" not in df.columns:
            if "comment" in df.columns:
                df["clean_comment"] = df["comment"].apply(clean_text)
            else:
                st.error("No 'comment' column found in uploaded CSV.")
                st.stop()

        st.write("### Sample Comments Loaded")
        st.write(df.head())

        run_analysis = st.button("Run Sentiment Analysis")
        if run_analysis:
            sentiments, scores = [], []
            progress_bar = st.progress(0)
            status_text = st.empty()

            comments_list = df["clean_comment"].tolist()
            total_comments = len(comments_list)

            for i, comment in enumerate(comments_list):
                status_text.text(f"Processing comment {i+1}/{total_comments}...")
                if pd.isnull(comment) or str(comment).strip() == "":
                    sentiments.append("neutral")
                    scores.append(0.0)
                else:
                    label, score = predict_sentiment(comment)
                    sentiments.append(label)
                    scores.append(score)
                progress_bar.progress((i + 1) / total_comments)

            df["sentiment"] = sentiments
            df["score"] = scores

            status_text.text("Sentiment analysis completed!")
            progress_bar.empty()

            st.write("### Comments with Sentiment")
            st.write(df.head(10))

            st.write("### Sentiment Distribution")
            sentiment_counts = df["sentiment"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.axis("equal")
            st.pyplot(fig2)

            st.write("### Word Cloud of Comments")
            text_combined = " ".join(df["clean_comment"].dropna().tolist())
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(text_combined)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

            st.download_button(
                "Download Results CSV",
                data=df.to_csv(index=False),
                file_name="youtube_comments_with_sentiment.csv",
                mime="text/csv",
            )
