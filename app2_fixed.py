
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

# ✅ Set page config FIRST — before any other Streamlit function
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎭", layout="centered")

# --- Load Resources ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gru_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# --- Utility Functions ---
def clean_text(text: str) -> str:
    """Clean input text using same preprocessing as training."""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def predict_sentiment(review: str):
    """Predict sentiment for a given review."""
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    proba = model.predict(padded)[0]
    classes = ['Negative', 'Neutral', 'Positive']
    return classes[np.argmax(proba)], proba, classes

# --- Streamlit UI ---
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #4CAF50;'>🎭 Movie Sentiment Analyzer</h1>
        <p style='font-size: 18px;'>Paste a movie review below and get an AI-powered sentiment analysis!</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("sentiment_form"):
    review = st.text_area("💬 Enter your movie review:", height=150, placeholder="Type something like 'The movie was absolutely amazing!'")
    submitted = st.form_submit_button("🔍 Analyze Sentiment")

if submitted:
    if not review.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        label, confidence, classes = predict_sentiment(review)

        # Display Result
        color_map = {'Negative': '#FF6B6B', 'Neutral': '#FFA500', 'Positive': '#4CAF50'}
        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {color_map[label]}; color: white;'>
                <h2>Predicted Sentiment: {label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display Confidence Scores
        st.markdown("### 📊 Confidence Breakdown:")
        for i, cls in enumerate(classes):
            raw_score = confidence[i]
            percent = raw_score * 100
            st.markdown(f"**{cls}: {percent:.2f}%**")
            st.progress(float(raw_score))

        # Optional footer
st.markdown(
    """
    <hr style="margin-top: 30px;">
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made with ❤️ using Streamlit and TensorFlow | GRU Model
    </div>
    """,
    unsafe_allow_html=True
)
