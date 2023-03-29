#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    #lower case
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and punctuation
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into text
    text = ' '.join(tokens)
    return text

# load vectorizer from disk
vectorizer = joblib.load('vectorizer2.pkl')
# load the model from disk
model = joblib.load('sentimentanalyse.pkl')

st.markdown("<h1 style='text-align: center;'>Sentment Analyse</h1>", unsafe_allow_html=True)

st.header('Enter Your Review of Titanic and Make a Sentiment Prediction ')
article = st.text_input('')
st.write(article)


processed_article = clean_text(article)

X_article = vectorizer.transform([processed_article])

if st.button('Predict'):
    result = model.predict(X_article)

    if result[0] == 1:
        
        st.write('Your review is classified as Positive review')
    else:
        st.write('Your review is classified as Negative review')


st.header('Sample Text')
st.markdown('''Oh my goodness, I just saw the most incredible film. It was so amazing that I couldn't stop yawning throughout the entire thing! The acting was just superb, with the lead actress delivering her lines in a monotone voice that made me feel like I was watching paint dry. And the special effects were so impressive, I could hardly tell they were fake.The plot was also so gripping, I couldn't help but doze off a few times. I loved how the story meandered aimlessly for hours, with no discernible direction or purpose. It was like watching a snail race - slow, uneventful, and thoroughly unexciting.Overall, I would highly recommend this film to anyone who enjoys being bored out of their mind. It's a true masterpiece of tedium, and I can't wait to watch it again and again and again... or maybe just once more, if I can stay awake.''')






