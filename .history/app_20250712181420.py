import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud


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
