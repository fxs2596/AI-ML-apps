# === START OF FILE: basic_rag.py ===
# Save this code as: C:\Users\siddi\AI_ML_Portfolio\pandas_knowledge_base\basic_rag.py

import os
import sys
import nltk
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer # For embeddings
from sklearn.metrics.pairwise import cosine_similarity # For similarity search
# Updated import for openai>=1.0.0
from openai import OpenAI # Import the client class

# --- Configuration ---
# Path to your folder containing the text files for the knowledge base
# Using '.' to refer to the current directory since script and txt files are expected in the same folder
knowledge_base_path = r"." # <--- Using '.' for the current directory

# Chunking parameters
chunk_size = 500 # Number of characters per chunk
chunk_overlap = 100 # Number of overlapping characters between chunks

# --- IMPORTANT: Your OpenAI API Key ---
# Read the API key from an environment variable for security
# Set the OPENAI_API_KEY environment variable in your terminal before running this script
llm_api_key = os.getenv("OPENAI_API_KEY") # <--- Read key from environment variable


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


# --- Define the text cleaning function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    try:
        words = nltk.word_tokenize(text)
    except LookupError:
        print("NLTK 'punkt' tokenizer not found during script runtime. Ensure NLTK data is downloaded.")
        return ""

    try:
        words = [word for word in words if word not in stop_words]
    except NameError:
         print("Stopwords not defined. Ensure 'stopwords' were downloaded.")
         pass

    return ' '.join(words)


# --- Load Documents ---
print(f"Loading documents from {knowledge_base_path}...")
documents = []
# Adjusted path joining since knowledge_base_path is '.'
try:
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(".txt"):
            # Construct full path using knowledge_base_path (.) and filename
            file_path = os.path.join(knowledge_base_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
except FileNotFoundError: # Catch error if '.' is not a valid path (shouldn't happen here)
    print(f"Error: Knowledge base path '{knowledge_base_path}' not found.")
    sys.exit(1)


if not documents:
    print("No text files found in the knowledge base path. Please check the path and file extensions.")
    print("Ensure your .txt files are directly in the pandas_knowledge_base folder if knowledge_base_path is '.'")
    sys.exit(1)

print(f"Loaded {len(documents)} documents.")

# Combine all documents into a single string for simple chunking
all_text = "\n".join(documents)


# --- Chunk Text ---
print("\nChunking text...")
chunks = []
start = 0
while start < len(all_text):
    end = start + chunk_size
    chunk = all_text[start:end]
    chunks.append(chunk)
    start += chunk_size - chunk_overlap

print(f"Created {len(chunks)} chunks.")


# --- Generate Embeddings for Chunks ---
print("\nGenerating embeddings for chunks...")

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-V2')
    print("Sentence Transformer model loaded.")
except Exception as e:
     print(f"Error loading Sentence Transformer model: {e}")
     print("Please ensure you have internet access or the model is cached.")
     sys.exit(1)


try:
    chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    print(f"Generated embeddings with shape: {chunk_embeddings.shape}")
except Exception as e:
    print(f"Error generating chunk embeddings: {e}")
    sys.exit(1)

# Store chunks and their embeddings together - This is our in-memory index
indexed_knowledge_base = list(zip(chunks, chunk_embeddings))

print(f"Indexed knowledge base with {len(indexed_knowledge_base)} items.")


# --- Build the Retriever ---
def get_query_embedding(query, model):
    """Generates an embedding for the user's query."""
    if query is None or not isinstance(query, str) or not query.strip():
         return None
    try:
        query_embedding = model.encode([query])[0]
        return query_embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def find_similar_chunks(query_embedding, indexed_kb, top_k=3):
    """Finds the top K most similar chunks to the query embedding."""
    if query_embedding is None or not indexed_kb:
        return []

    chunk_embeddings = [item[1] for item in indexed_kb]
    chunk_texts = [item[0] for item in indexed_kb]

    try:
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        return []

    actual_top_k = min(top_k, len(indexed_kb))
    if actual_top_k <= 0:
        return []

    top_k_indices = similarities.argpartition(-actual_top_k)[-actual_top_k:]
    top_k_indices = top_k_indices[(-similarities[top_k_indices]).argsort()]

    top_chunks = [(chunk_texts[i], similarities[i]) for i in top_k_indices]

    return top_chunks


# --- Build the Generator (LLM Integration - Updated for openai>=1.0.0) ---
# Create an OpenAI client instance outside the function for efficiency
# The API key is passed when creating the client
try:
    # Use the llm_api_key read from environment variable
    client = OpenAI(api_key=llm_api_key)
    # Optional: Test a simple API call here to confirm key is valid
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}], max_tokens=10)
    # print("OpenAI API key seems valid.")
    can_generate = True # Flag set if client initialization is successful
except Exception as e:
    print(f"\nError initializing OpenAI client with provided key: {e}")
    print("Please check your API key or ensure the OPENAI_API_KEY environment variable is set correctly.")
    can_generate = False
    client = None # Ensure client is None if initialization fails


def generate_answer(query, retrieved_chunks, openai_client): # Takes client as argument
    """Generates an answer to the query using the retrieved chunks and an LLM (OpenAI)."""

    if not retrieved_chunks:
        return "I couldn't find any relevant information in the knowledge base to answer your question."
    if not openai_client:
         return "LLM client is not initialized."


    context = "\n---\n".join([chunk for chunk, score in retrieved_chunks])

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based *only* on the provided context. If the answer cannot be found in the context, respond with 'I cannot answer this question based on the provided information.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    try:
        response = openai_client.chat.completions.create( # Use the client instance
            model="gpt-3.5-turbo", # Or "gpt-4" if you have access and prefer
            messages=prompt_messages,
            max_tokens=300, # Limit the answer length
            temperature=0.0 # Keep temperature low for factual answers based on context
        )
        answer = response.choices[0].message.content.strip()

        return answer

    except Exception as e:
        print(f"Error calling OpenAI LLM API: {e}")
        return f"An error occurred while generating the answer: {e}"


# --- Main RAG Interaction Loop ---
print("\n--- Basic RAG System (Giant Panda Knowledge Base) ---")
print("Type your questions about giant pandas.")
print("Type 'quit' or 'exit' to stop.")


# Ensure embedding_model is loaded
if 'embedding_model' not in locals() or embedding_model is None:
     print("\nEmbedding model not loaded. Cannot run RAG system.")
     sys.exit(1)

# The OpenAI client initialization and check is done above the function definitions


print("\nEmbedding model and knowledge base loaded. Ready to answer questions.")

while True:
    query = input("\nYou: ")
    if query.lower() in ["quit", "exit"]:
        break

    # 1. Retrieve relevant chunks
    retrieved_chunks = find_similar_chunks(
        get_query_embedding(query, embedding_model),
        indexed_knowledge_base,
        top_k=5 # Get top 5 chunks for context
    )

    if not retrieved_chunks:
        print("\nAI: I couldn't find any relevant information in the knowledge base to answer that.")
        continue

    # 2. Generate answer using the LLM (only if client is successfully initialized)
    if can_generate and client:
         answer = generate_answer(query, retrieved_chunks, client) # Pass the client instance
         # 3. Print the answer
         print(f"\nAI: {answer}")
    else:
         print("\nAI: LLM generation is not available.")
         if llm_api_key is None:
              print("Reason: OPENAI_API_KEY environment variable is not set.")
         elif not can_generate or client is None:
              print("Reason: OpenAI client failed to initialize (check API key or network).")

         print("Retrieved chunks:")
         for chunk, score in retrieved_chunks:
              print(f"--- Chunk (Similarity: {score:.4f}) ---\n{chunk[:200]}...")
              print("-" * 10)


print("\nExiting Basic RAG system. Goodbye!")

# === END OF FILE: basic_rag.py ===