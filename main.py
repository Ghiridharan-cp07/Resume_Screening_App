import os
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from config import STATIC_DIR, TEMPLATES_DIR, SUPPORTED_FORMATS
from screening import process_screening, load_models, extract_text_from_bytes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models gracefully on startup
    load_models()
    yield
    # Cleanup on shutdown

app = FastAPI(title="Resume Screening App", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/screen")
async def screen_resumes(
    jd_file: UploadFile = File(...),
    resume_files: List[UploadFile] = File(...),
    shortlist_size: int = Form(10)
):
    try:
        # 1. Process JD
        jd_ext = os.path.splitext(jd_file.filename)[1].lower()
        if jd_ext not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported JD format: {jd_ext}")
            
        jd_bytes = await jd_file.read()
        jd_text = extract_text_from_bytes(jd_bytes, jd_ext)
        if not jd_text.strip():
            raise HTTPException(status_code=400, detail="Job Description has no extractable text.")
            
        # 2. Process Resumes
        resumes_data = []
        for r_file in resume_files:
            r_ext = os.path.splitext(r_file.filename)[1].lower()
            if r_ext not in SUPPORTED_FORMATS:
                continue # Skip unsupported
                
            r_bytes = await r_file.read()
            r_text = extract_text_from_bytes(r_bytes, r_ext)
            if not r_text.strip():
                continue
                
            resumes_data.append({
                "file_name": r_file.filename,
                "text": r_text
            })
            
        if not resumes_data:
            raise HTTPException(status_code=400, detail="No valid resumes found with extractable text.")
            
        # 3. Call Screening Pipeline
        results = process_screening(jd_text, resumes_data, shortlist_size)
        
        return JSONResponse(content=results)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
