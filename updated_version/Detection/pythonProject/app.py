import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load your CSV file
csv_file_path = r'C:\Users\aslam\Detection\spam.csv'

# Reading the CSV file (assuming it has a "label" column for ham/spam and a "message" column for text)
data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Select the relevant columns
data = data[['v1', 'v2']]  # Assuming 'v1' is the label column ('ham'/'spam') and 'v2' is the message column
data.columns = ['label', 'message']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)

# Create a pipeline combining CountVectorizer, TfidfTransformer, and a Naive Bayes classifier
model = make_pipeline(CountVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, stop_words='english'),
                      TfidfTransformer(),
                      MultinomialNB(alpha=0.1))

# Train the model using the training set
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Streamlit app
st.title("Email Spam Detection")

# Input for the message
message = st.text_area("Enter the message to check ham/spam")

if st.button("Check!"):
    if message:
        # Predict whether the message is spam or ham
        prediction = model.predict([message])

        # Display the result: either "spam" or "ham"
        st.write(f"This message is: {prediction[0]}")  # Directly displays "spam" or "ham"
    else:
        st.write("Please enter a message to classify.")
