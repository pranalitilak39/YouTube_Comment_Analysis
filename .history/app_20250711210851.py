import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# 🌟 Load pretrained RoBERTa sentiment model
@st.cache_resource  # Cache the model to prevent reloading every time
def load_model():
    classifier = pipeline(
        "sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    return classifier


classifier = load_model()

# 🌟 Page title
st.title("YouTube Comment Sentiment Analysis using RoBERTa Transformer")
st.write(
    "🔬 **Research-based implementation using transformer NLP models (RoBERTa)** for advanced sentiment analysis."
)

# 🌟 Upload cleaned YouTube comments CSV
uploaded_file = st.file_uploader("Upload Cleaned YouTube Comments CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Comments Loaded")
    st.write(df.head())

    # 🌟 Sentiment Prediction Button
    if st.button("Run Sentiment Analysis"):
        sentiments = []
        scores = []

        # Process each comment
        for comment in df["clean_comment"]:
            if pd.isnull(comment) or comment.strip() == "":
                sentiments.append("neutral")
                scores.append(0.0)
            else:
                result = classifier(str(comment))
                label = result[0]["label"]
                score = result[0]["score"]
                sentiments.append(label)
                scores.append(score)

        # Add predictions to dataframe
        df["sentiment"] = sentiments
        df["score"] = scores

        # 🌟 Display final table
        st.write("### Comments with Sentiment")
        st.write(df.head(10))

        # 🌟 Sentiment Distribution Chart
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
