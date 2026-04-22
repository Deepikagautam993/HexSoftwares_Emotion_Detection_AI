import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------- Load Data --------------------

train_data = pd.read_csv("train.txt", sep=';', names=["text", "emotion"])
test_data = pd.read_csv("test.txt", sep=';', names=["text", "emotion"])

# -------------------- Split Data --------------------

X_train = train_data["text"]
y_train = train_data["emotion"]

X_test = test_data["text"]
y_test = test_data["emotion"]

# -------------------- Vectorization --------------------

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# -------------------- Model Training --------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

print("Model training completed!")

# -------------------- Evaluation --------------------

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

# -------------------- Custom Prediction --------------------

test_sentence = ["I am feeling very happy today"]

test_vector = vectorizer.transform(test_sentence)

prediction = model.predict(test_vector)

print("\nPredicted Emotion:", prediction[0])

# -------------------- Save Model --------------------

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")

# -------------------- Plot Confusion Matrix --------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()