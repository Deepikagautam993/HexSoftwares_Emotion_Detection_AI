import streamlit as st
import pickle
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="centered"
)

# ---------------- Load Model ----------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- Emotion Mapping ----------------
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲"
}

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Styling ----------------
st.markdown("""
<style>
.main-title {
    font-size: 52px;
    font-weight: bold;
    text-align: center;
    color: white;
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    margin-bottom: 10px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.02);}
    100% {transform: scale(1);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("About Project")

st.sidebar.info("""
Emotion Detection using ML & NLP.

Features:
- Emotion Prediction
- Confidence Score
- Emoji Output
- History Tracking
- Download Results
""")

st.sidebar.success("Model Loaded Successfully")

# ⭐ Supported Emotions List (IMPORTANT)
st.sidebar.subheader("Supported Emotions")

for emotion in emotion_emojis:
    st.sidebar.write(f"- {emotion}")

# ---------------- Main UI ----------------
st.markdown('<div class="main-title">Emotion Detection AI 😊</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today"
)

# ---------------- Prediction ----------------
if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    elif len(user_input.split()) < 3:
        st.warning("Please enter at least 3 words.")

    else:
        text_vector = vectorizer.transform([user_input])

        prediction = model.predict(text_vector)[0]

        # confidence
        prob = model.predict_proba(text_vector)
        confidence = max(prob[0]) * 100

        emoji = emotion_emojis.get(prediction, "🙂")

        st.success(f"Predicted Emotion: {prediction} {emoji}")
        st.info(f"Confidence: {confidence:.2f}%")

        # history
        st.session_state.history.append({
            "Text": user_input,
            "Emotion": prediction,
            "Confidence": f"{confidence:.2f}%"
        })

# ---------------- History ----------------
if st.session_state.history:
    st.subheader("Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download History CSV",
        data=csv,
        file_name="emotion_history.csv",
        mime="text/csv"
    )

# ---------------- Examples ----------------
st.subheader("Try Examples")

examples = [
    "I am very happy today",
    "I feel sad and lonely",
    "I am angry about this situation"
]

for ex in examples:
    if st.button(ex):
        vec = vectorizer.transform([ex])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)
        conf = max(prob[0]) * 100
        emoji = emotion_emojis.get(pred, "🙂")

        st.success(f"{pred} {emoji}")
        st.info(f"Confidence: {conf:.2f}%")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Emotion Detection AI Project | Built with ML & NLP")