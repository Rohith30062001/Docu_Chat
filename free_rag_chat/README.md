# DocuChat — Free Local RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) app:
- Upload PDF, TXT, Markdown, DOCX, CSV and XLSX files
- Extract text and split it into overlapping chunks
- Generate embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`
- Store vectors persistently in local ChromaDB
- Retrieve the most relevant chunks for a question
- Generate an answer locally using Ollama
- Display retrieved source chunks so you can inspect what the model used

## Architecture

`Files -> Text extraction -> Chunking -> Local embeddings -> ChromaDB -> Query embedding -> Similarity search -> Retrieved text -> Ollama chat model -> Answer + sources`

## Requirements

- Python 3.10 or 3.11 recommended
- Ollama installed on your Mac
- Internet for the first download of Python packages and models
- Disk space for the embedding model and chat model

## 1. Create a virtual environment

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails on your Mac, use Python 3.11 in a fresh virtual environment.

## 2. Install and start Ollama

Install Ollama from the official site: https://ollama.com

Then in Terminal:

```bash
ollama pull llama3.2:3b
ollama serve
```

If Ollama is already running as a macOS app, you may not need to run `ollama serve`.

In another terminal, test the model:

```bash
ollama run llama3.2:3b
```

The first model download can be several GB. If your Mac has limited free RAM or disk space, choose a smaller model available in Ollama and enter its exact name in the app's sidebar.

## 3. Run the app

From the project directory, with the virtual environment active:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Use it

1. Open the local URL in your browser.
2. Upload a small PDF/TXT/DOCX/CSV/XLSX file.
3. Click **Index uploaded files**.
4. Open **Chat with documents** and ask a question.
5. Expand **Retrieved source chunks** to inspect the text used for the answer.

## Where data is stored

- ChromaDB's persistent database: `./chroma_data/`
- Uploaded files are processed in memory/temp files; the app does not keep a copy of the original upload.
- Embedding model files are cached by Hugging Face/Sentence Transformers.

## Important limitations in this first version

- Chunking is character-based, not token-aware.
- CSV/Excel sheets are converted to text; this is useful for simple questions but is not a full analytical SQL engine.
- Scanned PDFs need OCR; this version extracts selectable PDF text only.
- The app is intended for local learning, not multi-user production deployment.
- If you change the embedding model, clear/rebuild the index so all vectors use the same embedding space.
- The first model downloads require internet, but inference runs locally afterward.
- Local generation is free of API charges, but uses your computer's memory/CPU/GPU.

## Optional hosted APIs

You can later add a hosted embedding API or hosted chat model behind a provider interface. Free hosted quotas and model availability can change; keep API keys in environment variables or Streamlit secrets, never commit them to Git. The initial build deliberately uses local inference so the project remains usable without paid API credits.

## Next engineering upgrades

1. Token-aware chunking and page numbers in source metadata
2. Source citations and relevance thresholds
3. Conversation-aware retrieval
4. Reranking with a cross-encoder
5. Embedding/model selection in the UI
6. Hosted-model adapter (for example Hugging Face Inference Providers)
7. Authentication, per-user collections, and document deletion
8. Evaluation set with Recall@k and answer faithfulness checks
