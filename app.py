import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model("model/emotion_model.keras")
tokenizer = joblib.load("model/tokenizer.pkl")

emotion_map = {
    0: "😢 Sadness",
    1: "😊 Joy",
    2: "❤️ Love",
    3: "😠 Anger",
    4: "😨 Fear",
    5: "😲 Surprise"
}

st.title("🧠 Emotion Detection using Bi-LSTM")
st.write("Enter a sentence and detect its emotion.")

text = st.text_area("Enter Text")

if st.button("Predict"):
    if text.strip():
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=50, padding="post")

        prediction = model.predict(padded, verbose=0)

        emotion = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Emotion: {emotion_map[emotion]}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text.")