import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


# Create a DataFrame from the data
df = pd.read_excel('data.xlsx')

print(df.isna().sum())
df.dropna(inplace=True)

X = df['text']
y = df[['label_price', 'label_service', 'label_convenience']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_vectorized = vectorizer.transform(X_test)

# Create a multi-output classifier with logistic regression
clf = MultiOutputClassifier(LogisticRegression())

# Train the classifier
clf.fit(X_train_vectorized, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_vectorized)

# Evaluate the classifier for each label
labels = ['label_price', 'label_service', 'label_convenience']
for i, label in enumerate(labels):
    print(f"{label} Classification Report:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Export the model and vectorizer
joblib.dump(clf, 'sentiment_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')