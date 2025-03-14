import streamlit as st
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer


model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))

st.title("🎬 IMDb Sentiment Analysis")
user_review = st.text_input('Enter your review')

if st.button('Predict'):
    if user_review:
       
        review_tfidf = vectorizer.transform([user_review]).toarray()
        
       
        result = model.predict(review_tfidf)

        
        if result[0] == 1:
            st.write('😊 Positive Review')
        else:
            st.write('😡 Negative Review')
    else:
        st.warning("Please enter a review.")


   