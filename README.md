# Domain-Specific Chatbot Using RAG for Healthcare FAQs

A domain-specific Retrieval-Augmented Generation (RAG) chatbot for healthcare FAQs. It uses TF-IDF + cosine similarity for retrieval and FLAN-T5-base for generation, with a CLI loop and a Streamlit UI.

## Features
- TF-IDF retrieval with bigrams and cosine similarity
- Open-source FLAN-T5-base text2text generation
- Topic-tagged knowledge base with optional filtering
- Similarity threshold fallback for low-confidence answers
- Streamlit UI with sources and experiments dashboard
- CLI interactive loop for quick testing

## Tech Stack
- Python 3.8+
- scikit-learn (TfidfVectorizer, cosine_similarity)
- HuggingFace Transformers (pipeline text2text-generation)
- PyTorch backend (via transformers)
- Streamlit UI

## Project Structure
rag_chatbot/
  app.py
  chatbot.py
  healthcare_data.txt
  requirements.txt

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   pip install -r rag_chatbot/requirements.txt

## Run the Streamlit App
streamlit run rag_chatbot/app.py

## Run the CLI Chatbot
python rag_chatbot/chatbot.py

## Notes
- The knowledge base is stored in healthcare_data.txt with paragraphs separated by blank lines.
- Each paragraph can start with a Topic: header for filtering in the UI.

## License
MIT
