import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # <-- Added the fix for the NLTK Lookup Error
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# ==========================================
# LOAD MODELS & RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    # Load TF-IDF Vectorizer and Logistic Regression Model
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    lr_model = joblib.load('logistic_regression_model.pkl')
    
    # 1. Manually recreate the exact architecture
    bilstm_model = Sequential([
        Embedding(input_dim=50000, output_dim=128, input_length=100),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])
    
    # 2. Build the model architecture to allocate memory
    bilstm_model.build(input_shape=(None, 100))
    
    # 3. Load ONLY the raw numerical weights
    if os.path.exists("bilstm_sentiment_model.keras"):
        bilstm_model.load_weights("bilstm_sentiment_model.keras")
    elif os.path.exists("bilstm_sentiment_model.h5"):
        bilstm_model.load_weights("bilstm_sentiment_model.h5")
    
    # Load Tokenizer
    try:
        tokenizer = joblib.load('tokenizer.pkl')
    except FileNotFoundError:
        tokenizer = None
        
    return tfidf_vectorizer, lr_model, bilstm_model, tokenizer

tfidf_vectorizer, lr_model, bilstm_model, tokenizer = load_resources()

# ==========================================
# TEXT CLEANING FUNCTION
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(words)

# Configuration
label_mapping = {0: 'Irrelevant', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}
MAX_LEN = 100 

# ==========================================
# STREAMLIT UI INTERFACE
# ==========================================
st.set_page_config(page_title="Twitter Sentiment Dashboard", page_icon="🐦", layout="centered")

st.title("🐦 Twitter Sentiment Analysis")
st.markdown("Enter a tweet below to analyze its sentiment using your trained Machine Learning models.")

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose Inference Engine:", ["Logistic Regression (TF-IDF)", "BiLSTM (Deep Learning)"])

if model_choice == "BiLSTM (Deep Learning)" and tokenizer is None:
    st.sidebar.warning("⚠️ `tokenizer.pkl` not found! Please save the tokenizer in Colab to use the BiLSTM.")

user_tweet = st.text_area("Enter Tweet Here:", placeholder="E.g., I am incredibly happy with this new update!")

if st.button("Predict Sentiment"):
    if not user_tweet.strip():
        st.error("Please enter a valid text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned_tweet = clean_text(user_tweet)
            
            if model_choice == "Logistic Regression (TF-IDF)":
                vector = tfidf_vectorizer.transform([cleaned_tweet])
                prediction_idx = lr_model.predict(vector)[0]
                sentiment = label_mapping[prediction_idx]
                
            elif model_choice == "BiLSTM (Deep Learning)":
                if tokenizer is None:
                    st.error("Cannot proceed without Tokenizer. Falling back to Logistic Regression...")
                    vector = tfidf_vectorizer.transform([cleaned_tweet])
                    prediction_idx = lr_model.predict(vector)[0]
                    sentiment = label_mapping[prediction_idx]
                else:
                    seq = tokenizer.texts_to_sequences([cleaned_tweet])
                    pad = pad_sequences(seq, maxlen=MAX_LEN)
                    pred_dl = bilstm_model.predict(pad)
                    prediction_idx = np.argmax(pred_dl, axis=1)[0]
                    sentiment = label_mapping[prediction_idx]
            
            # Display Results
            st.markdown("### Analysis Result:")
            
            if sentiment == "Positive":
                st.success(f"**Sentiment:** {sentiment} 😃")
            elif sentiment == "Negative":
                st.error(f"**Sentiment:** {sentiment} 😡")
            elif sentiment == "Neutral":
                st.info(f"**Sentiment:** {sentiment} 😐")
            else:
                st.warning(f"**Sentiment:** {sentiment} 🤷‍♂️")
                
            with st.expander("Show Cleaned Text"):
                st.write(cleaned_tweet)