# Customer Review Sentiment Analysis Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Review Sentiment Dashboard",
    page_icon="📱",
    layout="wide"
)

# LOAD DATASET
df = pd.read_excel("dataset -P667.xlsx")

# LOAD MODEL & VECTORIZER

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# TEXT CLEANING
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# CREATE SENTIMENT LABELS

def get_sentiment(rating):
    if rating in [4, 5]:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# FEATURE ENGINEERING

df["sentiment"] = df["rating"].apply(get_sentiment)

# Combine title and body
# Full review gives richer customer feedback

df["full_review"] = df["title"].astype(str) + " " + df["body"].astype(str)

# Clean reviews

df["clean_review"] = df["full_review"].apply(clean_text)

# Review length features

df["char_count"] = df["body"].astype(str).apply(len)
df["word_count"] = df["body"].astype(str).apply(lambda x: len(x.split()))

# SIDEBAR MENU

st.sidebar.title("📌 Navigation Menu")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Project Overview",
        "📂 Dataset",
        "📊 EDA Visualizations",
        "🤖 Model Performance",
        "🧠 Live Prediction"
    ]
)

# PAGE 1 - PROJECT OVERVIEW

if page == "🏠 Project Overview":

    st.title("📱 Customer Review Sentiment Analysis Dashboard")

    st.markdown("""
    ### 📌 Project Objective
    This project analyzes customer mobile reviews and predicts sentiment using Machine Learning and NLP techniques.
    """)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(df))
    col2.metric("Positive Reviews", (df['sentiment'] == 'Positive').sum())
    col3.metric("Negative Reviews", (df['sentiment'] == 'Negative').sum())
    col4.metric("Best Accuracy", "77.78%")

    st.markdown("---")

    st.subheader("✅ Workflow Completed")

    st.markdown("""
    - Data Cleaning
    - Exploratory Data Analysis
    - NLP Preprocessing
    - TF-IDF Vectorization
    - Logistic Regression Model
    - Naive Bayes Model
    - SVM Model
    - Model Comparison
    - Streamlit Dashboard Deployment
    """)

    st.info("Best Performing Model: Logistic Regression")

# PAGE 2 - DATASET

elif page == "📂 Dataset":

    st.title("📂 Dataset Information")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Dataset Columns")
    st.write(df.columns.tolist())

    st.subheader("Sample Dataset")
    st.dataframe(df.head(10))

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(info_df)

# PAGE 3 - EDA VISUALIZATIONS

elif page == "📊 EDA Visualizations":

    st.title("📊 Exploratory Data Analysis")

    # Rating Distribution
    st.subheader("⭐ Rating Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='rating', data=df, palette='Blues', ax=ax)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    st.write("Most reviews are 5-star ratings, while 1-star reviews are also significant.")

    # Sentiment Distribution
    st.subheader("😊 Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        x='sentiment',
        data=df,
        order=['Positive', 'Neutral', 'Negative'],
        palette='Set2',
        ax=ax
    )

    st.pyplot(fig)

    st.write("Positive reviews dominate the dataset, showing overall customer satisfaction.")

    # Word Count Distribution
    st.subheader("📝 Review Length Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['word_count'], bins=30, kde=True, color='purple', ax=ax)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.write("Most customer reviews contain around 20–60 words.")

    # WordCloud
    st.subheader("☁️ WordCloud of Reviews")

    text = " ".join(df['clean_review'])

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.write("Battery, camera, quality, and price are major discussion topics.")

    # Top Frequent Words
    st.subheader("📌 Top 15 Frequent Words")

    all_words = " ".join(df['clean_review']).split()
    word_freq = Counter(all_words)

    top_words = pd.DataFrame(
        word_freq.most_common(15),
        columns=['Word', 'Frequency']
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=top_words, palette='viridis', ax=ax)
    st.pyplot(fig)

# PAGE 4 - MODEL PERFORMANCE

elif page == "🤖 Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    result_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Naive Bayes', 'SVM'],
        'Accuracy': [0.7778, 0.7118, 0.7743]
    })

    st.subheader("📋 Model Accuracy Table")
    st.dataframe(result_df)

    st.subheader("📊 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x='Accuracy',
        y='Model',
        data=result_df,
        palette='coolwarm',
        ax=ax
    )

    ax.set_xlim(0.65, 0.80)
    st.pyplot(fig)

    st.success("🏆 Best Performing Model: Logistic Regression")

    st.write("Logistic Regression achieved the highest accuracy of 77.78%.")

# PAGE 5 - LIVE PREDICTION

elif page == "🧠 Live Prediction":

    st.title("🧠 Live Sentiment Prediction")

    st.write("Enter a customer review below to predict sentiment.")

    review = st.text_area(
        "✍️ Enter Review Text",
        height=150,
        placeholder="Example: The camera quality is amazing and battery backup is excellent."
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:
            cleaned = clean_text(review)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]

            st.subheader("Prediction Result")

            if pred == "Positive":
                st.success("😊 Positive Review")

            elif pred == "Negative":
                st.error("😞 Negative Review")

            else:
                st.info("😐 Neutral Review")

    st.markdown("---")

    st.subheader("📌 Try Sample Reviews")

    st.write("✅ Excellent battery life and amazing camera quality.")
    st.write("❌ Worst phone ever, poor display and heating issue.")
    st.write("➖ Average product, camera is okay but battery normal.")


# FOOTER
st.markdown("---")
st.caption("Built using Streamlit | NLP + Machine Learning Project")
