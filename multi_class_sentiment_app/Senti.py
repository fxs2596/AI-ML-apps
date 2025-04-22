import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib
import sys

# --- Configuration ---
# Your CSV file path for the multi-class data
csv_file_path = r"C:\Users\siddi\AI_ML_Portfolio\multi_class_sentiment_app\imdb_top_1000.csv"
text_column = 'Overview'
rating_column = 'IMDB_Rating'

# --- Download necessary NLTK data (only need to do this once per environment) ---
print("Checking for NLTK data (if not already present)...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found, attempting download...")
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK 'punkt': {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'punkt': {e}")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' not found, attempting download...")
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK 'stopwords': {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'stopwords': {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
     print("NLTK 'punkt_tab' not found, attempting download...")
     try:
         nltk.download('punkt_tab', quiet=True)
     except Exception as e:
         print(f"Error downloading NLTK 'punkt_tab': {e}")
except Exception as e:
    print(f"An error occurred checking for NLTK 'punkt_tab': {e}")

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
     print("Stopwords not found. Ensure 'stopwords' were downloaded.")
     stop_words = set()


# --- Load Data ---
print(f"Loading data from {csv_file_path}...")
try:
    try:
        df = pd.read_csv(csv_file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='latin1')
    print("Data loaded successfully.")
    print(f"Original data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    if text_column not in df.columns or rating_column not in df.columns:
         raise ValueError(f"Required columns ('{text_column}' and '{rating_column}') not found in the CSV.")
    df.dropna(subset=[text_column, rating_column], inplace=True)
    print(f"Shape after dropping rows with missing text/rating: {df.shape}")
    df[rating_column] = pd.to_numeric(df[rating_column], errors='coerce')
    df.dropna(subset=[rating_column], inplace=True)
    print(f"Shape after ensuring numeric rating and dropping NaNs: {df.shape}")

    print(f"\nSample data head (showing '{text_column}' and '{rating_column}'):\n", df[[text_column, rating_column]].head())

except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    print("Please check the file path and try again.")
    sys.exit(1)
except ValueError as ve:
    print(f"Data loading error: {ve}")
    print("Please check the column names and ensure the rating column can be converted to numbers.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading the CSV: {e}")
    print("Please check the file format.")
    sys.exit(1)

# --- Multi-class Sentiment Labeling based on Rating Ranges ---
print(f"\nCreating multi-class sentiment labels based on '{rating_column}' ranges...")

def create_sentiment_label(rating):
    if rating >= 9.0:
        return 'Excellent'
    elif rating >= 8.5:
        return 'Great'
    elif rating >= 8.0:
        return 'Very Good'
    elif rating >= 7.5:
        return 'Good'
    elif rating >= 7.0:
        return 'One Time Watch!'
    else:
        return 'Needs Improvement'

df['sentiment_category'] = df[rating_column].apply(create_sentiment_label)

print(f"Value counts for created sentiment categories:\n{df['sentiment_category'].value_counts()}")

initial_class_counts = df['sentiment_category'].value_counts()
classes_to_keep = initial_class_counts[initial_class_counts > 0].index.tolist()
if len(classes_to_keep) < len(initial_class_counts):
    df = df[df['sentiment_category'].isin(classes_to_keep)].copy()
    print(f"\nValue counts after removing classes with 0 samples:\n{df['sentiment_category'].value_counts()}")
    y_filtered = df['sentiment_category']
else:
     y_filtered = df['sentiment_category']


# --- Multi-class Sentiment Labeling based on Rating Ranges ---
# ... (your existing labeling code and value counts print) ...

# --- Plot Class Distribution --- # <--- ADD THIS SECTION
print("\nPlotting sentiment class distribution...")
plt.figure(figsize=(8, 6))
df['sentiment_category'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Sentiment Categories (Multi-Class)')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.tight_layout() # Adjust layout to prevent labels overlapping
# Save the plot
plt.savefig('multi_class_sentiment_distribution.png')
print("Sentiment class distribution plot saved.")
# plt.show() # Uncomment to display the plot immediately when running the script
# --- Text Preprocessing ---
print("\nPerforming text preprocessing on the 'Overview' column...")
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
         print("NLTK 'punkt' tokenizer not found during training script. Ensure NLTK data is downloaded.")
         return ""

    try:
        words = [word for word in words if word not in stop_words]
    except NameError:
         print("Stopwords not defined. Ensure 'stopwords' were downloaded.")
         pass

    return ' '.join(words)

if text_column not in df.columns:
    print(f"Error: Text column '{text_column}' not found after loading.")
    sys.exit(1)
df['cleaned_overview'] = df[text_column].apply(clean_text)

print("Text cleaning complete.")
print("\nSample cleaned overview:\n", df['cleaned_overview'].iloc[0][:200], "...")

# --- Data Splitting ---
print("\nSplitting data into training and testing sets...")
X = df['cleaned_overview']
y = y_filtered

if len(y.unique()) < 2:
    print("\nWarning: Only one sentiment class found after labeling/filtering.")
    print("Cannot perform stratified split or multi-class classification effectively.")
    print("Consider using a different dataset or adjusting rating ranges if possible.")
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.20, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- TF-IDF Vectorization ---
print("\nPerforming TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"X_train_tfidf shape (after vectorization): {X_train_tfidf.shape}")
print(f"X_test_tfidf shape (after vectorization): {X_test_tfidf.shape}")
print("TF-IDF vectorization complete.")

# --- Model Training (Support Vector Classifier) ---
if len(y.unique()) < 2:
     print("\nSkipping model training due to only one sentiment class being present.")
else:
    print("\nTraining Support Vector Classifier (SVC) model with balanced class weights...")
    model = SVC(kernel='linear', class_weight='balanced', probability=True)

    model.fit(X_train_tfidf, y_train)

    print("Model training complete.")

    # --- Model Evaluation (for Multi-class) ---
    print("\nEvaluating the model on the test set...")

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    try:
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\nModel Evaluation Metrics (Multi-class):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
    except ValueError as ve:
         print(f"Could not calculate weighted metrics: {ve}. This can happen with very few samples per class in the test set.")
         print(f"Accuracy: {accuracy:.4f}")

    print("\nInitial multi-class model training and evaluation complete.")

# --- Model Evaluation (for Multi-class) ---
# ... (your existing evaluation metrics print) ...

print("\nGenerating Confusion Matrix and Classification Report...") # <--- ADD THIS SECTION

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(
    model, X_test_tfidf, y_test,
    display_labels=model.classes_, # Use the model's classes
    cmap=plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title('Confusion Matrix (Multi-Class SVC)')
plt.tight_layout()
# Save the plot
plt.savefig('multi_class_confusion_matrix.png')
print("Confusion Matrix plot saved.")
# plt.show() # Uncomment to display the plot

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=model.classes_, zero_division=0))

print("\nVisualization and Reporting complete.")

# --- Save Model and Vectorizer ---
# ... (your existing saving code) ...
# --- Save Model and Vectorizer ---
if len(y.unique()) >= 2:
    print("\nSaving the trained model and vectorizer...")

    model_filename = 'svc_model.joblib'
    vectorizer_filename = 'tfidf_vectorizer.joblib'

    if 'model' in locals():
        try:
            joblib.dump(model, model_filename)
            print(f"Trained model saved to {model_filename}")
        except Exception as e:
             print(f"Error saving model: {e}")
    else:
        print("\nModel was not trained due to single class, skipping save.")

    if 'vectorizer' in locals():
        try:
            joblib.dump(vectorizer, vectorizer_filename)
            print(f"Fitted vectorizer saved to {vectorizer_filename}")
        except Exception as e:
             print(f"Error saving vectorizer: {e}")
    else:
        print("\nVectorizer object not found, skipping save.")

    print("\nModel and vectorizer save process complete.")
else:
    print("\nSkipping model and vectorizer save due to only one sentiment class being present.")