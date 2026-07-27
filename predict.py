import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("model/emotion_model.keras")
tokenizer = joblib.load("model/tokenizer.pkl")

emotion_map = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😠",
    4: "Fear 😨",
    5: "Surprise 😲"
}

while True:
    text = input("\nEnter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        break

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding="post")

    prediction = model.predict(padded, verbose=0)

    emotion = prediction.argmax()

    confidence = prediction.max() * 100

    print(f"\nEmotion: {emotion_map[emotion]}")
    print(f"Confidence: {confidence:.2f}%")