import re
import io
import fitz
from docx import Document
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import (
    EMBEDDING_MODEL_PATH, NER_MODEL_PATH, EMBEDDING_DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP, JD_QUERY_PREFIX,
    SEMANTIC_WEIGHT, EXPERIENCE_WEIGHT, NER_CONFIDENCE_THRESHOLD,
    PREPROCESSING_CONFIG
)

# Global Model Variables (Loaded Once)
_embedding_model = None
_ner_pipeline = None

def load_models():
    """Load ML models globally so they aren't loaded per request."""
    global _embedding_model, _ner_pipeline
    
    if _embedding_model is None:
        print(f"Loading embedding model from {EMBEDDING_MODEL_PATH} on {EMBEDDING_DEVICE}...")
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=EMBEDDING_DEVICE)
        except Exception as e:
            # Fallback if local path is just BAAI/bge-large-en-v1.5 from json
            print(f"Failed to load from local path, trying direct name: {e}")
            _embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=EMBEDDING_DEVICE)

    if _ner_pipeline is None:
        print(f"Loading NER model from {NER_MODEL_PATH}...")
        device_id = 0 if torch.cuda.is_available() and EMBEDDING_DEVICE == "cuda" else -1
        try:
            _ner_pipeline = pipeline(
                "ner",
                model=NER_MODEL_PATH,
                tokenizer=NER_MODEL_PATH,
                aggregation_strategy="simple",
                device=device_id
            )
        except Exception as e:
            print(f"Failed to load NER from local path: {e}")
            _ner_pipeline = pipeline(
                "ner",
                model="yashpwr/resume-ner-bert",
                tokenizer="yashpwr/resume-ner-bert",
                aggregation_strategy="simple",
                device=device_id
            )


# --- 1. File Extraction ---
def extract_text_from_bytes(file_bytes, extension):
    if extension == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document)
    if extension == ".docx":
        document = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if extension == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    return ""

# --- 2. Preprocessing ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    if PREPROCESSING_CONFIG.get("lowercase", True):
        text = text.lower()
        
    text = text.replace("\t", " ")
    
    if PREPROCESSING_CONFIG.get("replace_bullets", True):
        text = re.sub(r"[•‣▪◦●·∙]", "-", text)
        
    if PREPROCESSING_CONFIG.get("remove_urls", True):
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        
    allowed_chars = PREPROCESSING_CONFIG.get("allowed_characters", "a-z0-9+#./-,()&@:")
    # We must escape the allowed chars properly for regex or just use the notebook's raw regex.
    # Notebook's exact logic:
    text = re.sub(r"[^a-z0-9+#./\-,()&@:\n ]", " ", text)
    
    if PREPROCESSING_CONFIG.get("collapse_spaces", True):
        text = re.sub(r"[ ]{2,}", " ", text)
        
    if PREPROCESSING_CONFIG.get("preserve_line_breaks", True):
        text = "\n".join(line.strip() for line in text.split("\n"))
        
    if PREPROCESSING_CONFIG.get("collapse_excess_newlines", True):
        text = re.sub(r"\n{3,}", "\n\n", text)
        
    return text.strip()

# --- 3. Section Splitting ---
SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "objective", "career objective", "about me"},
    "skills": {"skills", "technical skills", "core skills", "key skills", "technical expertise", "technologies", "skills & technologies"},
    "experience": {"experience", "work experience", "professional experience", "employment history", "work history"},
    "projects": {"projects", "academic projects", "personal projects", "key projects", "project experience"},
    "education": {"education", "academic background", "educational background", "qualifications"},
    "certifications": {"certifications", "certificates", "licenses"},
    "achievements": {"achievements", "accomplishments", "awards", "honors"}
}
SECTION_LOOKUP = {alias.lower(): section for section, aliases in SECTION_ALIASES.items() for alias in aliases}

def detect_resume_sections(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    detected = []
    
    for index, line in enumerate(lines):
        normalized = re.sub(r"[^a-zA-Z0-9& ]", "", line).strip().lower()
        if normalized in SECTION_LOOKUP:
            detected.append((index, SECTION_LOOKUP[normalized]))
            
    sections = {}
    for position, (line_index, section_name) in enumerate(detected):
        start = line_index + 1
        end = detected[position + 1][0] if position + 1 < len(detected) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if content:
            sections[section_name] = content
            
    if len(sections) < 2:
        sections = {"full_text": text.strip()}
    return sections

JD_HEADING_MAP = {
    "role": ["job title", "title", "role", "position", "designation", "about the role", "role overview", "job summary", "about the job", "overview"],
    "required_skills": ["required skills", "requirements", "must have", "must haves", "required qualifications", "skills required", "key skills", "technical skills", "what we are looking for", "who you are", "essential skills", "mandatory skills", "technical requirements"],
    "preferred_skills": ["preferred skills", "nice to have", "good to have", "bonus", "preferred qualifications", "desirable", "plus points", "added advantage"],
    "responsibilities": ["responsibilities", "key responsibilities", "what you will do", "what you'll do", "duties", "job description", "your role", "day to day"],
    "experience": ["experience", "experience required", "work experience", "years of experience", "eligibility"],
    "education": ["education", "qualification", "qualifications", "educational qualification", "academic requirements"],
    "company": ["about us", "about the company", "who we are", "company overview"],
    "benefits": ["benefits", "what we offer", "perks", "compensation", "salary"]
}
DROP_SECTIONS = {"company", "benefits"}

def clean_heading(line):
    line = line.strip().lower()
    line = re.sub(r"^[#*\-\d\.\)\s]+", "", line)
    line = re.sub(r"\s*\([^)]*\)", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" :-")

def identify_section(line):
    heading = clean_heading(line)
    for section, variants in JD_HEADING_MAP.items():
        if heading in variants:
            return section
    return None

def identify_inline_section(line):
    raw = line.strip()
    if ":" not in raw:
        return None, ""
    head, value = raw.split(":", 1)
    section = identify_section(head)
    if section is None:
        return None, ""
    return section, value.strip()

def split_jd_sections(text):
    sections = {}
    current_section = None
    current_lines = []
    
    def flush():
        if current_section is None or current_section in DROP_SECTIONS:
            return
        content = "\n".join(current_lines).strip()
        if content:
            sections[current_section] = content
            
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        if set(line) <= set("-=_#* "): continue
        
        inline_section, inline_value = identify_inline_section(line)
        if inline_section is not None:
            flush()
            current_section = inline_section
            current_lines = [inline_value] if inline_value else []
            continue
            
        section = identify_section(line)
        if section is not None:
            flush()
            current_section = section
            current_lines = []
            continue
            
        if current_section is not None:
            current_lines.append(line)
            
    flush()
    return {sec: content for sec, content in sections.items() if sec not in DROP_SECTIONS and content}

# --- 4. Chunking & Embeddings ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words: return []
    if len(words) <= chunk_size: return [text.strip()]
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        if end == len(words): break
        start = end - overlap
    return chunks

def prepare_sections(sections):
    prepared = {}
    for section_name, content in sections.items():
        chunks = chunk_text(content)
        if chunks:
            prepared[section_name] = chunks
    return prepared

def generate_jd_embeddings(jd_chunks):
    inputs = []
    records = []
    index = 0
    
    for section_name, chunks in jd_chunks.items():
        for chunk_index, chunk in enumerate(chunks):
            inputs.append(f"{JD_QUERY_PREFIX}{chunk}")
            records.append({
                "section": section_name,
                "chunk_index": chunk_index,
                "text": chunk
            })
            
    if not inputs:
        return []
        
    embeddings = _embedding_model.encode(
        inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    
    for i, rec in enumerate(records):
        rec["embedding"] = embeddings[i]
        
    return records

def generate_resume_embeddings(resume_chunks):
    section_embeddings = {}
    for section_name, chunks in resume_chunks.items():
        embeddings = _embedding_model.encode(
            chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        section_embeddings[section_name] = embeddings
    return section_embeddings

# --- 5. Matching & Scoring ---
SECTION_MATCH_MAP = {
    "required_skills": ["skills", "experience", "projects"],
    "experience": ["experience", "summary"],
    "responsibilities": ["experience", "projects"],
    "preferred_skills": ["skills", "projects", "experience"]
}

def match_resume_to_jd(resume_sections, resume_embeddings, jd_records):
    matches = []
    for jd_record in jd_records:
        jd_section = jd_record["section"]
        jd_embedding = jd_record["embedding"]
        resume_sections_to_search = SECTION_MATCH_MAP.get(jd_section, list(resume_embeddings.keys()))
        
        candidate_matches = []
        for resume_section in resume_sections_to_search:
            if resume_section not in resume_embeddings:
                continue
                
            similarities = jd_embedding @ resume_embeddings[resume_section].T
            for chunk_index, score in enumerate(similarities):
                candidate_matches.append({
                    "jd_section": jd_section,
                    "jd_chunk_index": jd_record["chunk_index"],
                    "resume_section": resume_section,
                    "resume_chunk_index": chunk_index,
                    "score": float(score),
                    "jd_text": jd_record["text"],
                    "resume_text": resume_sections[resume_section][chunk_index]
                })
                
        if candidate_matches:
            best_match = max(candidate_matches, key=lambda item: item["score"])
            matches.append(best_match)
    return matches

SECTION_WEIGHTS = {
    "required_skills": 0.40,
    "experience": 0.30,
    "responsibilities": 0.20,
    "preferred_skills": 0.10
}

def calculate_section_scores(matches):
    section_scores = {}
    for section_name in SECTION_WEIGHTS:
        scores = [match["score"] for match in matches if match["jd_section"] == section_name]
        section_scores[section_name] = float(np.mean(scores)) if scores else 0.0
    return section_scores

def calculate_overall_score(section_scores):
    return float(sum(section_scores[sn] * w for sn, w in SECTION_WEIGHTS.items()))

EXPERIENCE_PATTERN = r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b"

def extract_years_of_experience(text):
    matches = re.findall(EXPERIENCE_PATTERN, text, flags=re.IGNORECASE)
    if not matches:
        return None
    values = [float(val) for val in matches]
    return max(values)

def calculate_experience_score(candidate_years, required_years):
    if required_years is None: return 1.0
    if candidate_years is None: return 0.0
    if candidate_years >= required_years: return 1.0
    return float(candidate_years / required_years)

# --- 6. Information Extraction ---
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN = r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)"
LINKEDIN_PATTERN = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9._%-]+"
GITHUB_PATTERN = r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9._-]+"

def get_header_text(text, max_lines=40, max_chars=1500):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines[:max_lines])[:max_chars]

def extract_contact_information(text):
    emails = re.findall(EMAIL_PATTERN, text, flags=re.IGNORECASE)
    phones = []
    for value in re.findall(PHONE_PATTERN, text):
        digits = re.sub(r"\D", "", value)
        if 10 <= len(digits) <= 13:
            phones.append(value.strip())
            
    linkedin = re.findall(LINKEDIN_PATTERN, text, flags=re.IGNORECASE)
    github = re.findall(GITHUB_PATTERN, text, flags=re.IGNORECASE)
    
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedin[0] if linkedin else None,
        "github": github[0] if github else None
    }

def extract_ner_information(text):
    try:
        entities = _ner_pipeline(text)
    except Exception:
        return {}
        
    extracted = {}
    for entity in entities:
        if entity["score"] < NER_CONFIDENCE_THRESHOLD:
            continue
        label = entity["entity_group"]
        value = entity["word"].replace(" ##", "").strip()
        if len(value) < 2:
            continue
            
        extracted.setdefault(label, [])
        if value not in extracted[label]:
            extracted[label].append(value)
    return extracted

def extract_name(ner_information, header_text):
    for label, values in ner_information.items():
        label_upper = label.upper()
        if "NAME" in label_upper or label_upper in {"PER", "PERSON"}:
            for value in values:
                words = value.split()
                if 2 <= len(words) <= 4:
                    return value.title()
                    
    header_lines = [line.strip() for line in header_text.split("\n") if line.strip()]
    for line in header_lines[:5]:
        if "@" in line or re.search(r"\d{4}", line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'-]+", word) for word in words):
            return line.title()
    return None

def extract_candidate_information(text):
    header = get_header_text(text)
    ner_information = extract_ner_information(header)
    contact_information = extract_contact_information(text)
    
    return {
        "name": extract_name(ner_information, header),
        "email": contact_information["email"],
        "phone": contact_information["phone"],
        "linkedin": contact_information["linkedin"],
        "github": contact_information["github"]
    }

# --- 7. Main Pipeline Runner ---
def generate_reason(semantic_score, experience_score, candidate_years, required_years):
    parts = []
    if semantic_score > 0.7:
        parts.append("Strong alignment with required skills and responsibilities")
    elif semantic_score > 0.5:
        parts.append("Moderate alignment with the role's technical requirements")
    else:
        parts.append("Limited alignment with the core requirements")
        
    if required_years:
        if candidate_years and candidate_years >= required_years:
            parts.append("meets experience requirements.")
        else:
            parts.append("falls short of experience requirements.")
    else:
        if candidate_years:
            parts.append("has relevant experience.")
        else:
            parts.append(".")
            
    return ", ".join(parts).capitalize().replace(".,", ".")

def process_screening(jd_text, resumes, shortlist_size):
    load_models()
    
    # JD Processing
    jd_clean = preprocess_text(jd_text)
    jd_sections = split_jd_sections(jd_clean)
    jd_chunks = prepare_sections(jd_sections)
    jd_records = generate_jd_embeddings(jd_chunks)
    jd_required_years = extract_years_of_experience(jd_text)
    
    results = []
    
    # Resume Processing
    for resume in resumes:
        text = resume["text"]
        clean_text = preprocess_text(text)
        sections = detect_resume_sections(clean_text)
        section_chunks = prepare_sections(sections)
        
        # Embeddings & Matches
        section_embeddings = generate_resume_embeddings(section_chunks)
        matches = match_resume_to_jd(section_chunks, section_embeddings, jd_records)
        
        # Scoring
        section_scores = calculate_section_scores(matches)
        semantic_score = calculate_overall_score(section_scores)
        
        cand_years = extract_years_of_experience(text)
        exp_score = calculate_experience_score(cand_years, jd_required_years)
        
        final_score = (SEMANTIC_WEIGHT * semantic_score) + (EXPERIENCE_WEIGHT * exp_score)
        
        # Info Extraction
        cand_info = extract_candidate_information(text)
        
        # Prepare Result
        results.append({
            "candidate_name": cand_info["name"] or resume["file_name"],
            "email": cand_info["email"],
            "phone": cand_info["phone"],
            "linkedin": cand_info["linkedin"],
            "github": cand_info["github"],
            "years_of_experience": cand_years,
            "fit_score": final_score,
            "semantic_score": semantic_score,
            "experience_score": exp_score,
            "reason": generate_reason(semantic_score, exp_score, cand_years, jd_required_years),
            "evidence": [
                {
                    "jd_section": m["jd_section"],
                    "resume_section": m["resume_section"],
                    "score": m["score"],
                    "match_text": m["resume_text"]
                } for m in sorted(matches, key=lambda x: x["score"], reverse=True)[:3]
            ],
            "file_name": resume["file_name"]
        })
        
    # Ranking
    results.sort(key=lambda x: x["fit_score"], reverse=True)
    
    # Normalizing Fit Scores (Min-Max) for display if needed, but the prompt says 
    # to preserve original values or display percentages. Notebook did Min-Max normalized_score.
    if len(results) > 0:
        score_min = min(r["fit_score"] for r in results)
        score_max = max(r["fit_score"] for r in results)
        for r in results:
            if score_max > score_min:
                norm = (r["fit_score"] - score_min) / (score_max - score_min)
            else:
                norm = 1.0
            r["normalized_score"] = norm
            # Convert to display percentage using normalized score, or just use fit_score directly?
            # Notebook ranked by normalized_score. Let's provide both or update fit_score to be normalized.
            r["display_score"] = round(norm * 100, 1)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "total_resumes": len(results),
        "shortlist_size": shortlist_size,
        "results": results[:shortlist_size]
    }
