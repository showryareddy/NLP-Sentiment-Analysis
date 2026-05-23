# Customer Review Sentiment Analysis Dashboard

import streamlit as st
import pickle

# Load files
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("logreg.pkl", "rb"))

st.title("Sentiment Analysis App")

text = st.text_area("Enter your review")

if st.button("Predict"):

    vector = tfidf.transform([text])

    prediction = model.predict(vector)

    st.success(f"Prediction: {prediction[0]}")
