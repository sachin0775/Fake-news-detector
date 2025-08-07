import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
import lime.lime_text

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop and len(word) > 2]
    return ' '.join(words)

# Load model and TF-IDF vectorizer
@st.cache_resource
def load_pickles():
    with open('model.pkl', 'rb') as m:
        model = pickle.load(m)
    with open('tfidf.pkl', 'rb') as t:
        tfidf = pickle.load(t)
    return model, tfidf

model, tfidf = load_pickles()
pipeline = make_pipeline(tfidf, model)
explainer = lime.lime_text.LimeTextExplainer(class_names=['REAL', 'FAKE'])

# Prediction function
def predict(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0].max()
    label = 'FAKE' if pred == 1 else 'REAL'
    return label, proba, cleaned

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector with Explainable AI")

st.sidebar.header("About the App")
st.sidebar.markdown("""
This app uses a machine learning model to detect fake news.
- Model: Logistic Regression (or calibrated)
- Explainer: LIME
- Vectorizer: TF-IDF
""")

user_input = st.text_area("Paste a news article or headline below:")

if st.button("Predict"):
    if user_input:
        label, proba, cleaned = predict(user_input)
        st.success(f"**Prediction:** {label}  \n**Confidence:** {proba:.2f}")
        st.markdown("### 🔍 LIME Explanation")
        exp = explainer.explain_instance(cleaned, pipeline.predict_proba, num_features=10)
        st.components.v1.html(exp.as_html(), height=400)

st.info("This app is for educational use. Class labels assumed: 1=FAKE, 0=REAL.")
