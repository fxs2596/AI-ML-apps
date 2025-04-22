import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression # Using Logistic Regression for binary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import joblib
import sys

# --- Configuration ---
# Your NEW CSV file path - Update this to the location of your new movie.csv file
csv_file_path = r"C:\Users\siddi\AI_ML_Portfolio\binary_sentiment_app\movie.csv" # <--- PATH TO YOUR movie.csv
text_column = 'text'
label_column = 'label'

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
    if text_column not in df.columns or label_column not in df.columns:
         raise ValueError(f"Required columns ('{text_column}' and '{label_column}') not found in the CSV.")
    df.dropna(subset=[text_column, label_column], inplace=True)
    print(f"Shape after dropping rows with missing text/label: {df.shape}")
    df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
    df.dropna(subset=[label_column], inplace=True)
    df[label_column] = df[label_column].astype(int)
    print(f"Shape after ensuring numeric label and dropping NaNs: {df.shape}")

    print(f"\nSample data head (showing '{text_column}' and '{label_column}'):\n", df[[text_column, label_column]].head())
    print(f"Value counts for labels:\n{df[label_column].value_counts()}")

except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    print("Please check the file path and try again.")
    sys.exit(1)
except ValueError as ve:
    print(f"Data loading error: {ve}")
    print("Please check the column names and ensure the label column can be converted to numbers (0 or 1).")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading the CSV: {e}")
    print("Please check the file format.")
    sys.exit(1)


# --- Set X and y for splitting ---
X = df[text_column]
y = df[label_column]

# --- Load Data ---
# ... (your existing data loading code and value counts print) ...

# --- Plot Class Distribution --- # <--- ADD THIS SECTION
print("\nPlotting sentiment class distribution...")
plt.figure(figsize=(6, 5))
df[label_column].value_counts().sort_index().plot(kind='bar', color=['salmon', 'lightgreen']) # Assuming 0/1 mapped to Neg/Pos
plt.title('Distribution of Sentiment Labels (Binary)')
plt.xlabel('Sentiment Label (0: Negative, 1: Positive)')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.tight_layout()
# Save the plot
plt.savefig('binary_sentiment_distribution.png')
print("Sentiment class distribution plot saved.")
# plt.show() # Uncomment to display the plot

# --- Text Preprocessing ---
print("\nPerforming text preprocessing on the 'text' column...")
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
df['cleaned_text'] = df[text_column].apply(clean_text)

print("Text cleaning complete.")
print("\nSample cleaned text:\n", df['cleaned_text'].iloc[0][:200], "...")


# --- Data Splitting ---
print("\nSplitting data into training and testing sets...")
X_split = df['cleaned_text']
y_split = y

if len(y_split.unique()) < 2:
    print("\nWarning: Only one class found after labeling/filtering.")
    print("Cannot perform stratified split or binary classification effectively.")
    print("Consider using a different dataset.")
    X_train, X_test, y_train, y_test = train_test_split(
         X_split, y_split, test_size=0.20, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_split, y_split, test_size=0.20, random_state=42, stratify=y_split
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

# --- Model Training (Logistic Regression for Binary) ---
if len(y_split.unique()) < 2:
     print("\nSkipping model training due to only one class being present.")
else:
    print("\nTraining Logistic Regression model for binary sentiment...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    model.fit(X_train_tfidf, y_train)

    print("Model training complete.")

    # --- Model Evaluation (for Binary) ---
    print("\nEvaluating the model on the test set...")

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    try:
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}. Ensure model supports predict_proba.")
        roc_auc = None

    print(f"\nModel Evaluation Metrics (Binary Classification):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Positive class): {precision:.4f}")
    print(f"Recall (Positive class): {recall:.4f}")
    print(f"F1-Score (Positive class): {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")

    print("\nInitial binary model training and evaluation complete.")

# --- Model Evaluation (for Binary) ---
# ... (your existing binary evaluation metrics print) ...

# Add plotting and reporting only if model was trained successfully
if 'model' in locals(): # Check if the model object was created
    print("\nGenerating Confusion Matrix, Classification Report, and ROC Curve...") # <--- ADD THIS SECTION

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 7))
    # For binary 0 and 1 labels, display_labels can be custom strings
    ConfusionMatrixDisplay.from_estimator(
        model, X_test_tfidf, y_test,
        display_labels=['Negative', 'Positive'], # Custom labels for 0 and 1
        cmap=plt.cm.Blues
    )
    plt.title('Confusion Matrix (Binary LR)')
    plt.tight_layout()
    # Save the plot
    plt.savefig('binary_confusion_matrix.png')
    print("Confusion Matrix plot saved.")
    # plt.show() # Uncomment to display the plot

    # Print Classification Report
    print("\nClassification Report:")
    # target_names can be custom strings for 0 and 1
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], zero_division=0))

    # Plot ROC Curve (if AUC was calculated)
    if 'roc_auc' in locals() and roc_auc is not None: # Check if roc_auc was calculated
         try:
             plt.figure(figsize=(7, 7))
             RocCurveDisplay.from_estimator(model, X_test_tfidf, y_test)
             plt.title('ROC Curve (Binary LR)')
             plt.grid(True)
             plt.tight_layout()
             # Save the plot
             plt.savefig('binary_roc_curve.png')
             print("ROC Curve plot saved.")
             # plt.show() # Uncomment to display the plot
         except Exception as e:
              print(f"Error plotting ROC Curve: {e}")

    print("\nVisualization and Reporting complete.")

else: # If model training was skipped
    print("\nSkipping visualization and reporting as model training was skipped.")

# --- Save Model and Vectorizer ---
# ... (your existing saving code) ...

# --- Save Model and Vectorizer ---
if len(y_split.unique()) >= 2:
    print("\nSaving the trained model and vectorizer...")

    model_filename = 'binary_sentiment_lr_v2.joblib'
    vectorizer_filename = 'tfidf_vectorizer_v2.joblib'

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
    print("\nSkipping model and vectorizer save due to only one class being present.")