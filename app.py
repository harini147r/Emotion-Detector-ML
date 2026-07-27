import streamlit as st
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Cache Model
# -------------------------------
@st.cache_resource
def load_resources():
    model = load_model("model/emotion_model.keras")
    tokenizer = joblib.load("model/tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_resources()

emotion_map = {
    0: "😢 Sadness",
    1: "😊 Joy",
    2: "❤️ Love",
    3: "😠 Anger",
    4: "😨 Fear",
    5: "😲 Surprise"
}

st.title("🧠 Emotion Detection using Bi-LSTM")
st.write("Enter a sentence below to predict its emotion.")

text = st.text_area("Enter Text")

if st.button("Predict"):

    if text.strip():

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=50, padding="post")

        prediction = model.predict(padded, verbose=0)[0]

        emotion = np.argmax(prediction)
        confidence = prediction[emotion] * 100

        st.success(f"**Emotion:** {emotion_map[emotion]}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        chart_data = {
            emotion_map[i]: float(prediction[i] * 100)
            for i in range(6)
        }

        st.bar_chart(chart_data)

    else:
        st.warning("Please enter some text.")