import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("dataset/train.txt", sep=";", names=["text", "label"])
val = pd.read_csv("dataset/val.txt", sep=";", names=["text", "label"])
test = pd.read_csv("dataset/test.txt", sep=";", names=["text", "label"])

emotion_map = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}

train["label"] = train["label"].map(emotion_map)
val["label"] = val["label"].map(emotion_map)
test["label"] = test["label"].map(emotion_map)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(train["text"])

X_train = tokenizer.texts_to_sequences(train["text"])
X_val = tokenizer.texts_to_sequences(val["text"])
X_test = tokenizer.texts_to_sequences(test["text"])

max_length = 50

X_train = pad_sequences(X_train, maxlen=max_length, padding="post")
X_val = pad_sequences(X_val, maxlen=max_length, padding="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post")

y_train = train["label"]
y_val = val["label"]
y_test = test["label"]

print("Training Shape :", X_train.shape)
print("Validation Shape :", X_val.shape)
print("Testing Shape :", X_test.shape)

print("\nFirst Sequence:")
print(X_train[0])

print("\nFirst Label:")
print(y_train.iloc[0])

model = Sequential([
    Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=50
    ),

    Bidirectional(
        LSTM(128)
    ),

    Dropout(0.5),

    Dense(64, activation="relu"),

    Dense(6, activation="softmax")
])

model.build(input_shape=(None, 50))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict(X_test)

predicted_labels = predictions.argmax(axis=1)
cm = confusion_matrix(y_test, predicted_labels)


from sklearn.metrics import classification_report

predictions = model.predict(X_test)
predicted_labels = predictions.argmax(axis=1)

print(classification_report(y_test, predicted_labels))

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_names,
    yticklabels=emotion_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig("accuracy.png", dpi=300, bbox_inches="tight")
plt.savefig("loss.png", dpi=300, bbox_inches="tight")

import os
import joblib

os.makedirs("model", exist_ok=True)

model.save("model/emotion_model.keras")
joblib.dump(tokenizer, "model/tokenizer.pkl")

print("✅ Model Saved Successfully!")       
