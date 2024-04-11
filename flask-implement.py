from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
clf = joblib.load('sentiment_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Map the predicted labels to their corresponding meanings
label_map = {
    'label_price': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'},
    'label_service': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'},
    'label_convenience': {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text']
        
        # Vectorize the input text
        input_vectorized = vectorizer.transform([input_text])
        
        # Make predictions using the loaded model
        predictions = clf.predict(input_vectorized)
        
        # Create a dictionary to store the predicted sentiment for each label
        sentiment = {}
        for i, label in enumerate(['label_price', 'label_service', 'label_convenience']):
            predicted_label = predictions[0][i]
            sentiment[label] = label_map[label][predicted_label]
        
        return render_template('result.html', sentiment=sentiment, input_text=input_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)