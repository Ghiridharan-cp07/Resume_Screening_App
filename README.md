# Resume Screening Application

This is a production-quality Resume Screening web application built on top of an existing, validated NLP pipeline. The application provides an elegant, responsive UI for recruiters to upload a Job Description and multiple resumes, and ranks the candidates based on semantic and experience matching.

## Architecture

- **Backend**: FastAPI
- **Frontend**: Vanilla HTML, CSS, JavaScript (No external frameworks)
- **ML Pipeline**: Uses `sentence-transformers` (BAAI/bge-large-en-v1.5) and Hugging Face `pipeline` (yashpwr/resume-ner-bert) for local inference.

### Important Note on Models

**No models are trained or fine-tuned by this application.** The application exclusively relies on the existing validated models provided in the `artifacts/` directory.

## Project Structure

```
Resume_Screening_App/
│
├── artifacts/
│   ├── embedding_model/ 
│   ├── ner_model/       
│   ├── config.json      
│   └── preprocessing_config.json 
│
├── static/
│   ├── style.css
│   └── app.js
│
├── templates/
│   └── index.html
│
├── config.py
├── main.py
├── screening.py
└── requirements.txt
```

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Features

- Independent processing of PDF, DOCX, and TXT files.
- Advanced section-based text extraction and chunking.
- Semantic matching using local dense embeddings.
- Zero-shot/Regex candidate information extraction (Name, Email, Phone, GitHub, LinkedIn).
- Experience level extraction and scoring.
- Beautiful, intuitive recruiter interface.
