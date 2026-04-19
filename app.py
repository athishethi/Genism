import streamlit as st
import gensim.downloader as api
import pandas as pd

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Word2Vec App",
    layout="centered"
)

st.title("Word Similarity App")

# -------------------------------
# Load Model (SAFE + CONTROLLED)
# -------------------------------
@st.cache_resource
def load_model():
    return api.load("glove-wiki-gigaword-50")  # even smaller → safer

try:
    model = load_model()
except Exception as e:
    st.error("Model failed to load. Try refreshing.")
    st.stop()

st.success("Model loaded successfully!")

# -------------------------------
# Similarity Section
# -------------------------------
st.subheader("Check Similarity")

word1 = st.text_input("Word 1", "good")
word2 = st.text_input("Word 2", "great")

if st.button("Compare"):
    try:
        score = model.similarity(word1, word2)
        st.write(f"Similarity Score: {score:.4f}")
    except:
        st.error("Word not found in vocabulary")

# -------------------------------
# Similar Words
# -------------------------------
st.subheader("Find Similar Words")

word = st.text_input("Enter word", "king")

if st.button("Find"):
    try:
        results = model.most_similar(word, topn=5)
        for w, s in results:
            st.write(f"{w} → {s:.4f}")
    except:
        st.error("Word not found")

# -------------------------------
# Dataset Upload (SAFE)
# -------------------------------
st.subheader("Upload Dataset")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write(df.head())
