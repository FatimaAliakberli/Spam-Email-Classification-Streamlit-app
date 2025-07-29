import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import spacy
import string
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()


model = joblib.load("spam_classifier.pkl")

# Preprocessing functions (same as your training pipeline)
def clean_test(s):
    for cs in s:
        if not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s):
    wordsList = s.split()
    k_length = 2
    resultList = [element for element in wordsList if len(element) > k_length]
    resultString = ' '.join(resultList)
    return resultString

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(filtered_words)

def preprocess(text):
    text = clean_test(text)
    text = remove_little(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text

def email_prediction(email_text):
    processed_text = preprocess(email_text)
    prediction = model.predict([processed_text])[0]
    return "HAM (Not Spam)" if prediction == 1 else "SPAM"

# Streamlit UI
def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="centered")
    st.title("📨 Spam Email Classifier")
    st.write("Enter the email content below:")

    user_input = st.text_area("Email Content", height=300)

    if st.button("Predict"):
        if user_input:
            result = email_prediction(user_input)
            st.success(f"✅ The email is classified as: **{result}**")
        else:
            st.error("⚠️ Please enter some text to classify.")

if __name__ == "__main__":
    main()
