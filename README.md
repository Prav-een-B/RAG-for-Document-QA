# RAG for Document Question Answering

This is a small project where I tried to build a simple **Retrieval-Augmented Generation (RAG)** pipeline for answering questions from long documents (like PDFs).

Large language models are good at generating answers, but they usually cannot read an entire long document directly because of context limits. The idea here is to first **retrieve the most relevant parts of the document** and then use those pieces as context to produce an answer.

The project was mainly built to understand how RAG systems work internally instead of relying on frameworks like LangChain.

---

## What the project does

The pipeline roughly works like this:

1. A PDF document is loaded.
2. The text is extracted and broken into smaller chunks.
3. Each chunk is converted into a vector embedding.
4. When a question is asked, the system searches for the most relevant chunks using cosine similarity.
5. The most relevant sentences are selected.
6. Those sentences are passed to an LLM which produces a final summarized answer.

So the language model is not guessing blindly — it is answering using the retrieved context.

---

## Project structure

```text
RAG-for-Document-QA
│
├── app/                # core logic for embeddings, retrieval etc.
│
├── run.py              # entry point to run the pipeline
├── requirements.txt    # python dependencies
└── README.md
```

---

## Setup

Clone the repository

```bash
git clone https://github.com/Prav-een-B/RAG-for-Docnument-Q-and-A.git
cd RAG-for-Docnument-Q-and-A
```

Create a virtual environment

```bash
python -m venv RAG
```

Activate it

Windows

```bash
RAG\Scripts\activate
```

Linux / macOS

```bash
source RAG/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Important configuration before running

### 1. Add your API key

The answer generation step uses an LLM through **OpenRouter**.

Before running the project, open:

```text
app/generator.py
```

and replace the API key with your own (line 5):

```python
api_key = "YOUR_OPENROUTER_API_KEY"
```

You can get a key from:

https://openrouter.ai

---

### 2. Set the document path

The pipeline currently expects a specific PDF file.

Edit the run.py where the document path is defined (line 22) and update it to your document:

```python
DOC_PATH = "path/to/your/document.pdf"
```

Make sure the file exists before running the pipeline.

---

### 3. Clear cached embeddings when changing documents

The project stores processed embeddings/chunks locally so that the document does not need to be reprocessed every time.

If you switch to a **different PDF**, you should clear the cached data first (DELETE THE ENTIRE FOLDER NAMED **CACHE**), otherwise the system may still use embeddings from the previous document.

---

## First run

The **first run may take a few minutes** because:

* the PDF is parsed
* the document is chunked
* embeddings are generated
* models may be downloaded

After this initial processing, later runs should be faster.

---

## Running the project

Run the script:

```bash
python run.py
```

Then you can ask questions about the document, for example:

```
What does the survey say about inflation?
```

The system retrieves relevant chunks from the document and generates a short answer based on them.

---

## Why I made this

I wanted to understand the mechanics of RAG systems from scratch.
Instead of using a full framework, this implementation focuses on the core ideas:

* document chunking
* embedding generation
* semantic retrieval
* grounding LLM answers in retrieved context

This project is still experimental and mainly intended as a learning exercise.

---

## Author

Praveen B
IISER Bhopal
