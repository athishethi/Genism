import streamlit as st
import gensim.downloader as api
import pandas as pd

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Word2Vec Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Word2Vec Semantic Similarity App")

# -------------------------------
# Load Word2Vec Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    return api.load("word2vec-google-news-300")

wv = load_model()

st.success("Word2Vec model loaded successfully!")

# -------------------------------
# Word Similarity Section
# -------------------------------
st.header("🔍 Check Word Similarity")

col1, col2 = st.columns(2)

with col1:
    word1 = st.text_input("Enter first word", "great")

with col2:
    word2 = st.text_input("Enter second word", "good")

if st.button("Compute Similarity"):
    try:
        similarity = wv.similarity(word1, word2)
        st.metric("Similarity Score", f"{similarity:.4f}")
    except KeyError:
        st.error("One or both words not in vocabulary!")

# -------------------------------
# Similar Words Section
# -------------------------------
st.header("📌 Find Similar Words")

input_word = st.text_input("Enter a word to find similar words", "king")

if st.button("Get Similar Words"):
    try:
        similar_words = wv.most_similar(input_word, topn=10)
        st.write("Top similar words:")
        for word, score in similar_words:
            st.write(f"{word} → {score:.4f}")
    except KeyError:
        st.error("Word not found in vocabulary!")

# -------------------------------
# Dataset Section
# -------------------------------
st.header("📰 Dataset Viewer")

@st.cache_data
def load_data():
    return pd.read_csv("fake_and_real_news.csv")

try:
    df = load_data()

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

except FileNotFoundError:
    st.warning("Dataset file 'fake_and_real_news.csv' not found. Please upload it.")

# -------------------------------
# Upload Option
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset Shape:", df.shape)
    st.dataframe(df.head())