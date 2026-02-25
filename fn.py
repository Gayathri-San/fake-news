import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("lstm_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("Fake News Detection")

text = st.text_area("Enter news text")

if st.button("Predict"):
    if text.strip() == "":
        st.write("Please enter text")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=300)  # use SAME maxlen as training

        pred = model.predict(padded)[0][0]

        if pred > 0.5:
            st.write("❌ Fake News")
        else:
            st.write("✅ Real News")