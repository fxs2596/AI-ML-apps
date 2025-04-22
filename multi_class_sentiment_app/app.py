from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import re
from nltk.corpus import stopwords
import sys

# --- Configuration ---
# Filenames for the saved model and vectorizer from senti.py
model_filename = 'svc_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

# --- Download necessary NLTK data for the app (if not already present) ---
print("Checking for NLTK data for app (if not already present)...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found for app, attempting download...")
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
         print(f"Error downloading NLTK 'punkt' for app: {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'punkt' for app: {e}")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' not found for app, attempting download...")
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK 'stopwords' for app: {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'stopwords' for app: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
     print("NLTK 'punkt_tab' not found for app, attempting download...")
     try:
        nltk.download('punkt_tab', quiet=True)
     except Exception as e:
         print(f"Error downloading NLTK 'punkt_tab' for app: {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'punkt_tab' for app: {e}")

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
     print("Stopwords not found. Ensure 'stopwords' were downloaded for app.")
     stop_words = set()


# --- Load the Saved Model and Vectorizer ---
print("Loading model and vectorizer...")
loaded_model = None
loaded_vectorizer = None

try:
    loaded_model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found in the same directory as app.py")
    print("Exiting due to missing model file.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading model: {e}")
    print("Exiting due to loading error.")
    sys.exit(1)

try:
    loaded_vectorizer = joblib.load(vectorizer_filename)
    print(f"Vectorizer '{vectorizer_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Vectorizer file '{vectorizer_filename}' not found in the same directory as app.py")
    print("Exiting due to missing vectorizer file.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading vectorizer: {e}")
    print("Exiting due to loading error.")
    sys.exit(1)


# --- Define the text cleaning function (same as in senti.py) ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        print("NLTK 'punkt' tokenizer not found during app runtime. Ensure NLTK data is downloaded.")
        return ""

    try:
        words = [word for word in words if word not in stop_words]
    except NameError:
         print("Stopwords not defined for app. Ensure 'stopwords' were downloaded.")
         pass

    return ' '.join(words)


# --- Initialize Flask App ---
app = Flask(__name__)

# --- Define a Route for the Home Page ---
@app.route('/')
def home():
    # This route renders the index.html template from the default 'templates' folder
    return render_template('index.html')


# --- Define a Route for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None or 'overview' not in data:
        return jsonify({'error': 'Invalid input: Missing overview text'}), 400

    overview_text = data.get('overview', '')

    if not overview_text or not overview_text.strip():
         print("\nReceived empty text for analysis.")
         return jsonify({'sentiment': 'Please enter some text to analyze.'}), 200


    print(f"\nReceived text for analysis: {overview_text[:100]}...")

    cleaned_text = clean_text(overview_text)
    print(f"Cleaned text: {cleaned_text[:100]}...")

    if not cleaned_text:
        print("\nText cleaning resulted in an empty string.")
        return jsonify({'sentiment': 'Could not process the input text.'}), 200


    if not loaded_vectorizer:
         print("Error: Vectorizer not loaded.")
         return jsonify({'error': 'Vectorizer not available'}), 500

    try:
        text_vectorized = loaded_vectorizer.transform([cleaned_text])
        print(f"Text vectorized. Shape: {text_vectorized.shape}")
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return jsonify({'error': f'Error during text vectorization: {e}'}), 500


    if not loaded_model:
        print("Error: Model not loaded.")
        return jsonify({'error': 'Model not available'}), 500

    try:
        prediction = loaded_model.predict(text_vectorized)
        sentiment_result = prediction[0]
        print(f"Prediction made: {sentiment_result}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Error during model prediction: {e}'}), 500

    return jsonify({'sentiment': sentiment_result})


# --- Run the Flask App ---
if __name__ == '__main__':
    # Running on default port 5000
    app.run(debug=True)