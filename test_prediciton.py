import joblib

# Load the trained model and vectorizer
clf = joblib.load('sentiment_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Input text to predict sentiment
input_text = input("Enter the text to predict sentiment: ")

# Vectorize the input text
input_vectorized = vectorizer.transform([input_text])
predictions = clf.predict(input_vectorized)
label_map = {
    'label_price': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'},
    'label_service': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'},
    'label_convenience': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
}

# Print the predicted sentiment for each label
print("Predicted Sentiment:")
for i, label in enumerate(['label_price', 'label_service', 'label_convenience']):
    predicted_label = predictions[0][i]
    sentiment = label_map[label][predicted_label]
    print(f"{label}: {sentiment}")