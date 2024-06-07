import os
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import streamlit as st


# Extract text from html file
def extract_text_from_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
            soup = BeautifulSoup(file, "html.parser")
            return soup.get_text(separator=" ")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


# Load data
data = []
labels = []
base_path = "C:\\Users\\HP\\Downloads\\data1"  # Adjust the path to your data folder

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            text = extract_text_from_html(file_path)
            if text:  # Only add if text was successfully extracted
                data.append(text)
                labels.append(category)

df = pd.DataFrame({"text": data, "label": labels})


# Feature Extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]


# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Save the model and vectorizer
with open("../Users/HP/Downloads/data1/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("../Users/HP/Downloads/data1/vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)


# Load the trained model and vectorizer
with open("../Users/HP/Downloads/data1/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("../Users/HP/Downloads/data1/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


def classify_html_file(html_content):
    text = BeautifulSoup(html_content, "html.parser").get_text(separator=" ")
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return text, prediction[0]


def load_sample_data():
    # Load sample data (dummy function, replace with actual data loading)
    return ["Sample HTML content for category 1", "Sample HTML content for category 2"], ["category1", "category2"]


# Streamlit App
st.title("HTML File Classifier")


# File uploader to accept multiple HTML files
uploaded_files = st.file_uploader("Choose HTML files...", type="html", accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        html_content = uploaded_file.read().decode("utf-8", errors='ignore')
        text, category = classify_html_file(html_content)
        results.append({"File Name": uploaded_file.name, "Category": category, "Extracted Text": text})

    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Option to display the classification report on the training set
    if st.checkbox("Show classification report on training set"):
        sample_texts, sample_labels = load_sample_data()
        sample_vectors = vectorizer.transform(sample_texts)
        sample_predictions = model.predict(sample_vectors)
        report = classification_report(sample_labels, sample_predictions, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

    # Option to show extracted text for each file
    if st.checkbox("Show extracted text for each file"):
        for index, row in results_df.iterrows():
            st.subheader(f"File: {row['File Name']}")
            st.write(f"**Category**: {row['Category']}")
            st.write("**Extracted Text**:")
            st.write(row["Extracted Text"])