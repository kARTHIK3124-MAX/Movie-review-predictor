import streamlit as st
import pickle as pk
import os
import subprocess

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "scikit-learn"])
    from sklearn.feature_extraction.text import TfidfVectorizer





model = pk.load(open('model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))


st.title("🎬 IMDb Sentiment Analysis")
user_review = st.text_input('Enter your review')

if st.button('Predict'):
    if user_review:
       
        review_tfidf = vectorizer.transform([user_review]).toarray()
        
        
        result = model.predict(review_tfidf)

       
        falling_emoji_script = """
        <style>
        @keyframes fall {
            0% { transform: translateY(-100px); opacity: 1; }
            100% { transform: translateY(500px); opacity: 0; }
        }
        .emoji {
            position: absolute;
            font-size: 30px;
            animation: fall 3s linear infinite;
        }
        </style>
        <div style="position: relative; height: 500px;">
        """

       
        if result[0] == 1:
            st.success('😊 Positive Review')
            falling_emoji_script += "".join([f'<span class="emoji" style="left:{i*10}%">😊</span>' for i in range(10)])
        else:
            st.error('😡 Negative Review')
            falling_emoji_script += "".join([f'<span class="emoji" style="left:{i*10}%">😡</span>' for i in range(10)])

        falling_emoji_script += "</div>"
        st.markdown(falling_emoji_script, unsafe_allow_html=True)
    else:
        st.warning("Please enter a review.")


   
