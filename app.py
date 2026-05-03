import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model & vectorizer
model = joblib.load("sentiment_log_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("🎬 Sentiment Analysis Ulasan Film")
st.write("Aplikasi sederhana untuk klasifikasi ulasan film (positif/negatif).")

# Input teks manual
user_input = st.text_area("Masukkan ulasan film:")

if st.button("Prediksi Sentimen"):
    if user_input:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        prob = model.predict_proba(input_tfidf)[0]

        if prediction == "positive":
            st.success(f"Prediksi: Positif (Probabilitas: {prob[1]:.2f})")
        else:
            st.error(f"Prediksi: Negatif (Probabilitas: {prob[0]:.2f})")

# Upload file CSV untuk batch prediksi
uploaded_file = st.file_uploader("Upload file CSV ulasan", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    reviews = data['review']
    reviews_tfidf = vectorizer.transform(reviews)
    preds = model.predict(reviews_tfidf)

    # Hitung distribusi
    pos_count = (preds == "positive").sum()
    neg_count = (preds == "negative").sum()

    # Visualisasi pie chart
    fig, ax = plt.subplots()
    ax.pie([pos_count, neg_count], labels=["Positive", "Negative"], autopct='%1.1f%%', colors=["green", "red"])
    st.pyplot(fig)
