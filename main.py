import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('punkt')


def preprocess_message(message):
    stop_words = set(stopwords.words('english'))
    words = message.lower().split()  # Split message into words by spaces
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)


def main():
    # Load the dataset
    df = pd.read_csv("spam.csv", encoding='latin-1')

    # Clean the dataset
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Label encoding
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Apply preprocessing to all messages
    df['message'] = df['message'].apply(preprocess_message)

    # Feature extraction: vectorizing the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])

    # Define target variable
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
