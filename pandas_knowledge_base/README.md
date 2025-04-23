# Basic RAG System: Giant Panda Knowledge Base

An end-to-end **command-line interface (CLI)** demonstrating a Retrieval-Augmented Generation (RAG) pipeline. Ask natural-language questions about giant pandas, and the system retrieves relevant knowledge-base excerpts to ground its answers.

---

## 🚀 Demo

1. **Clone the repo**  
   ```bash
   git clone https://github.com/fxs2596/AI-ML-apps.git
   cd AI-ML-apps/pandas_knowledge_base
   ```
2. **Install dependencies** (using Conda or venv)  
   ```bash
   conda install pandas scikit-learn nltk joblib sentence-transformers openai matplotlib
   # Download necessary NLTK data:
   python -c "import nltk; nltk.download('punkt stopwords punkt_tab')"
   ```
3. **Add your API key**  
   ```bash
   export OPENAI_API_KEY="YOUR_SECRET_KEY"
   ```
4. **Populate your knowledge base**  
   Place your `.txt` files (e.g. `giant_panda.txt`, `pandas_list.txt`, `pandas_world.txt`) in this folder.
5. **Run the CLI**  
   ```bash
   python basic_rag.py
   ```
6. **Interact!**  
   Type any question about giant pandas and see the system in action.

---

## ✨ Key Features

- **Document Loader & Chunker**  
  Splits each text file into overlapping character-based chunks for context windows.

- **Embedder & Index**  
  Uses `sentence-transformers` to encode each chunk, storing embeddings in memory.

- **Semantic Retriever**  
  Computes cosine similarity between your query embedding and indexed chunks to fetch the top matches.

- **LLM-Powered Generator**  
  Feeds retrieved snippets as context to OpenAI’s GPT-3.5-turbo (or similar) for coherent, grounded answers.

- **Simple CLI**  
  Minimal setup—just type questions and read answers right in your terminal.

---

## 📁 Project Structure

```
pandas_knowledge_base/
├── basic_rag.py         # Main script
├── README.md            # (This file)
└── panda_knowledge_base/
    ├── giant_panda.txt
    ├── pandas_list.txt
    └── pandas_world.txt
```

---

## 📚 Data Source

All `.txt` files are sourced from Wikipedia’s *Giant Panda* entries. Feel free to add direct URLs or additional documents to expand your knowledge base.

---

## 🧠 ML / AI Approach

1. **Problem**: Question Answering via RAG  
2. **Representation**: Sentence embeddings for text chunks  
3. **Retrieval**: Cosine similarity over embeddings  
4. **Generation**: OpenAI LLM synthesizes an answer using retrieved context  
5. **Core Concepts**:  
   - Chunking & Overlap  
   - Embedding generation  
   - Semantic search  
   - Prompt engineering  
   - LLM integration

---

## 🚧 Future Improvements

- Web UI (Flask, Streamlit, Gradio)  
- Support for PDF, DOCX, and other formats  
- Integration with a vector database (e.g., FAISS, Pinecone)  
- Experiment with alternative embedding or retrieval algorithms  
- Add conversational memory to maintain context over multiple queries  
- Implement evaluation metrics (e.g., retrieval precision, answer quality)

---

Enjoy exploring the world of giant pandas with your very own RAG system! 🐼

