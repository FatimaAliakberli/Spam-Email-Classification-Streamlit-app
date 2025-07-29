import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
import spacy
import string
import re
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


df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
column_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
df = df.drop(column_to_remove, axis=1)
df = df.drop_duplicates()


df.loc[df["v1"] == "spam", "Category"] = 0
df.loc[df["v1"] == "ham", "Category"] = 1
X = df['v2']
y = df['Category']


# Remove non_ASCII characters
def clean_test(s):
  for cs in s:
    if not cs in string.ascii_letters:
      s = s.replace(cs, ' ')
  return s.rstrip('\r\n')

# Function to remove the words that have length less than or equal to 2
def remove_little(s):
  wordsList = s.split()
  k_length = 2
  resultList = [element for element in wordsList if len(element) > k_length]
  resultString = ' '.join(resultList)
  return resultString

# Function to reduce the words to their base or dictionary form
def lemmatize_text(text):
  doc = nlp(text)
  lemmatized_text = " ".join([token.lemma_ for token in doc])
  return lemmatized_text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(filtered_words)

from wordcloud import STOPWORDS
STOPWORDS = STOPWORDS

df['v2'] = df['v2'].apply(lambda x: clean_test(x))
df['v2'] = df['v2'].apply(lambda x: remove_little(x))
df['v2'] = df['v2'].apply(lambda x: lemmatize_text(x))
df['v2'] = df['v2'].apply(lambda x: remove_stopwords(x))

dic_all = {}
def count_words(s):
  global dic_all
  wordsList = s.split()
  for w in wordsList:
    if not w in dic_all:
      dic_all[w]  = 1
    else:
      dic_all[w] += 1

      
dic_all = {}
df['v2'].apply(lambda x: count_words(x))
dic_all = sorted(dic_all.items(), key = lambda x:x[1], reverse = True)
df_new = pd.DataFrame(dic_all)
df_new.head(20)




pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
pipeline.fit(X, y)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring = 'accuracy')
print(cv_scores)
print(cv_scores.mean())

def email_prediction(msg):
    msg = clean_test(msg)
    msg = remove_little(msg)
    msg = lemmatize_text(msg)
    msg = remove_stopwords(msg)
    
    prediction = pipeline.predict([msg])
    if prediction[0] == 0:
        return "Spam"
    else:
        return "Ham"
    

def main():
    st.title("Spam Email Classifier")
    st.write("Enter the email content below:")
    
    user_input = st.text_area("Email Content", height=300)
    
    if st.button("Predict"):
        if user_input:
            result = email_prediction(user_input)
            st.success(f"The email is classified as: {result}")
        else:
            st.error("Please enter some text to classify.")

if __name__ == "__main__":
    main()
