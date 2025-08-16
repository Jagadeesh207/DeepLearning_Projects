import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence



word_index=imdb.get_word_index()
reversed_word_index={value:key for key,value in word_index.items()}

model=load_model('simle_rnn_imdb.h5')

def decoded_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-1,'?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=200)
    return padded_review




st.title("Imdb Movie Review Sentiment Analysis")
st.write("Enter a moviereview to classify Either Positive or Negative")

user_input=st.text_ara("Movie Review")

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment:{sentiment}')
    st.write(f"Prediction_score:{prediction}")

else:
    st.write("Please enter movie Review")






