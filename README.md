# HTML-File-Classifier

This Python script classifies HTML files into different categories using Natural Language Processing (NLP) techniques. It utilizes libraries such as BeautifulSoup for HTML parsing and Scikit-learn for text vectorization and classification.

## Overview

The HTML File Classifier performs the following tasks:

1. **Data Preparation**:
   - Reads HTML files from a specified directory.
   - Extracts text content from each HTML file.

2. **Feature Extraction**:
   - Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

3. **Model Training**:
   - Trains a Logistic Regression model using the extracted features to classify HTML files into respective categories.

4. **Model Evaluation**:
   - Prints the classification report, including precision, recall, and F1-score, on the test set.

5. **Model Persistence**:
   - Saves the trained model and TF-IDF vectorizer using pickle for future use.

6. **Streamlit App**:
   - Provides a user-friendly interface for uploading HTML files and obtaining classification results.
   - Offers options to display classification reports and extracted text for uploaded files.

## Usage

1. **Setup**:
   - Ensure Python and required libraries are installed (`pip install -r requirements.txt`).
   - Adjust the base path to your HTML data folder in the script.

2. **Training**:
   - Run the script to train the model on your HTML dataset.

3. **Model Persistence**:
   - Trained model and vectorizer will be saved as `model.pkl` and `vectorizer.pkl`, respectively.

4. **Streamlit App**:
   - Run the Streamlit app (`streamlit run your_script.py`) to launch the classifier interface.
   - Upload HTML files to classify and view results interactively.

## File Structure

- **YourScript.py**: Main Python script containing the classifier logic and Streamlit app.
- **requirements.txt**: List of Python dependencies.
- **README.md**: This file providing an overview of the project.

## Additional Notes

- Ensure HTML files are structured and contain meaningful content for accurate classification.
- Customize the script as needed for specific requirements or additional functionalities.
