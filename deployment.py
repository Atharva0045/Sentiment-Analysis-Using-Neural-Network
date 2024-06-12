from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained ML model and vectorizer
model_filename = 'savemodel.sav'
vectorizer_filename = 'vectorizer.pkl'

with open('savemodel.sav', 'rb') as m:
    loaded_model = pickle.load(m);

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None

    if request.method == 'POST':
        headline = request.form['headline']
        sentiment_result = predict_sentiment(headline)

    return render_template('index.html', result=sentiment_result)

def predict_sentiment(headline):
    # Preprocess headline using the loaded vectorizer
    X = loaded_vectorizer.transform([headline]).toarray()
    
    # Predict sentiment using the loaded model
    prediction = loaded_model.predict(X)
    
    # Convert prediction to sentiment label
    sentiment_labels = ["neutral", "positive", "negative"]
    predicted_sentiment = sentiment_labels[np.argmax(prediction)]
    
    return predicted_sentiment


if __name__ == '__main__':
    app.run(debug=True)