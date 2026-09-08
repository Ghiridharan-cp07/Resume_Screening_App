!pip install PyMuPDF python-docx

from google.colab import drive
import os
import fitz
import pandas as pd
from docx import Document
from pprint import pprint

drive.mount("/content/drive")

PROJECT_PATH = "/content/drive/MyDrive/Resume_Screening_Data/Data"
RESUME_PATH = os.path.join(PROJECT_PATH, "Resumes")
JD_PATH = "/content/drive/MyDrive/Resume_Screening_Data/Data/JD/JD2.txt"

SUPPORTED_FORMATS = [".pdf", ".docx", ".txt"]

print(f"Project path: {PROJECT_PATH}")
print(f"Resume path: {RESUME_PATH}")
print(f"JD path: {JD_PATH}")

def extract_text(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        with fitz.open(file_path) as document:
            return "\n".join(page.get_text() for page in document)

    if extension == ".docx":
        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()

    return ""

resumes = []

for file_name in sorted(os.listdir(RESUME_PATH)):
    file_path = os.path.join(RESUME_PATH, file_name)

    if not os.path.isfile(file_path):
        continue

    if os.path.splitext(file_name)[1].lower() not in SUPPORTED_FORMATS:
        continue

    text = extract_text(file_path)

    resumes.append({
        "file_name": file_name,
        "file_path": file_path,
        "text": text,
        "n_words": len(text.split())
    })

resume_df = pd.DataFrame(resumes)

print(f"Resumes loaded: {len(resume_df)}")
print(f"Columns: {list(resume_df.columns)}")

pprint(
    resume_df[["file_name", "text"]].head(1).to_dict("records"),
    width=120
)

if not os.path.isfile(JD_PATH):
    raise FileNotFoundError(f"JD file not found: {JD_PATH}")

jd_text = extract_text(JD_PATH)

print(f"Selected JD: {os.path.basename(JD_PATH)}")
print(f"Characters: {len(jd_text)}")

pprint(jd_text, width=120)

import re
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("\t", " ")
    text = re.sub(r"[•‣▪◦●·∙]", "-", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#./\-,()&@:\n ]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

resume_df["clean_text"] = resume_df["text"].apply(
    preprocess_text
)

jd_clean = preprocess_text(jd_text)

print(f"JD: {len(jd_clean.split())} words")
print(
    "RES:",
    resume_df["clean_text"].apply(
        lambda text: len(text.split())
    ).tolist()
)

SECTION_ALIASES = {
    "summary": {
        "summary", "professional summary", "profile", "objective",
        "career objective", "about me"
    },
    "skills": {
        "skills", "technical skills", "core skills", "key skills",
        "technical expertise", "technologies", "skills & technologies"
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment history", "work history"
    },
    "projects": {
        "projects", "academic projects", "personal projects",
        "key projects", "project experience"
    },
    "education": {
        "education", "academic background", "educational background",
        "qualifications"
    },
    "certifications": {
        "certifications", "certificates", "licenses"
    },
    "achievements": {
        "achievements", "accomplishments", "awards", "honors"
    }
}

SECTION_LOOKUP = {
    alias.lower(): section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

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

resume_df["sections"] = resume_df["text"].apply(detect_resume_sections)

section_counts = {}

for sections in resume_df["sections"]:
    for section_name in sections:
        section_counts[section_name] = section_counts.get(section_name, 0) + 1

print(f"Resumes processed: {len(resume_df)}")
print("Detected sections:")
pprint(section_counts)

if len(resume_df) > 0:
    print(f"\nSample resume: {resume_df.iloc[0]['file_name']}")
    pprint(resume_df.iloc[0]["sections"], width=120)

import re

JD_HEADING_MAP = {
    "role": [
        "job title", "title", "role", "position", "designation",
        "about the role", "role overview", "job summary",
        "about the job", "overview"
    ],
    "required_skills": [
        "required skills", "requirements", "must have", "must haves",
        "required qualifications", "skills required", "key skills",
        "technical skills", "what we are looking for", "who you are",
        "essential skills", "mandatory skills", "technical requirements"
    ],
    "preferred_skills": [
        "preferred skills", "nice to have", "good to have", "bonus",
        "preferred qualifications", "desirable", "plus points",
        "added advantage"
    ],
    "responsibilities": [
        "responsibilities", "key responsibilities", "what you will do",
        "what you'll do", "duties", "job description", "your role",
        "day to day"
    ],
    "experience": [
        "experience", "experience required", "work experience",
        "years of experience", "eligibility"
    ],
    "education": [
        "education", "qualification", "qualifications",
        "educational qualification", "academic requirements"
    ],
    "company": [
        "about us", "about the company", "who we are", "company overview"
    ],
    "benefits": [
        "benefits", "what we offer", "perks", "compensation", "salary"
    ]
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
        for variant in variants:
            if heading == variant:
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

        if not line:
            continue

        if set(line) <= set("-=_#* "):
            continue

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

    return {
        section: content
        for section, content in sections.items()
        if section not in DROP_SECTIONS and content
    }

jd_sections = split_jd_sections(jd_clean)

print("JD sections found:")
for section, text in jd_sections.items():
    print(f"{section:<20} {len(text.split()):>4} words")

print("\nJD section keys:")
print(list(jd_sections.keys()))

print("\nJD section previews:")
for section, text in jd_sections.items():
    print(f"\n[{section}]")
    print(text[:500])

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=EMBEDDING_DEVICE
)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]).strip())

        if end == len(words):
            break

        start = end - overlap

    return chunks

def prepare_sections(sections):
    prepared = {}

    for section_name, content in sections.items():
        chunks = chunk_text(content)

        if chunks:
            prepared[section_name] = chunks

    return prepared

jd_chunks = prepare_sections(jd_sections)

resume_df["section_chunks"] = resume_df["sections"].apply(prepare_sections)

jd_embedding_inputs = []

for section_name, chunks in jd_chunks.items():
    for chunk in chunks:
        jd_embedding_inputs.append(f"Represent this sentence for searching relevant passages: {chunk}")

jd_embeddings = embedding_model.encode(
    jd_embedding_inputs,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

resume_embedding_data = []

for sections in resume_df["section_chunks"]:
    section_embeddings = {}

    for section_name, chunks in sections.items():
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        section_embeddings[section_name] = embeddings

    resume_embedding_data.append(section_embeddings)

resume_df["section_embeddings"] = resume_embedding_data

print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Device: {EMBEDDING_DEVICE}")
print(f"JD chunks: {len(jd_embedding_inputs)}")
print(f"Embedding dimension: {jd_embeddings.shape[1]}")

SECTION_MATCH_MAP = {
    "required_skills": ["skills", "experience", "projects"],
    "experience": ["experience", "summary"],
    "responsibilities": ["experience", "projects"],
    "preferred_skills": ["skills", "projects", "experience"]
}

jd_embedding_records = []
jd_embedding_index = 0

for section_name, chunks in jd_chunks.items():
    for chunk_index, chunk in enumerate(chunks):
        jd_embedding_records.append({
            "section": section_name,
            "chunk_index": chunk_index,
            "text": chunk,
            "embedding": jd_embeddings[jd_embedding_index]
        })
        jd_embedding_index += 1

def match_resume_to_jd(resume_sections, resume_embeddings):
    matches = []

    for jd_record in jd_embedding_records:
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

resume_df["section_matches"] = resume_df.apply(
    lambda row: match_resume_to_jd(
        row["section_chunks"],
        row["section_embeddings"]
    ),
    axis=1
)

print(f"Resumes matched: {len(resume_df)}")

if len(resume_df) > 0:
    print(f"\nSample resume: {resume_df.iloc[0]['file_name']}")
    pprint(resume_df.iloc[0]["section_matches"][:5], width=140)

SECTION_WEIGHTS = {
    "required_skills": 0.40,
    "experience": 0.30,
    "responsibilities": 0.20,
    "preferred_skills": 0.10
}

def calculate_section_scores(matches):
    section_scores = {}

    for section_name in SECTION_WEIGHTS:
        scores = [
            match["score"]
            for match in matches
            if match["jd_section"] == section_name
        ]

        section_scores[section_name] = float(np.mean(scores)) if scores else 0.0

    return section_scores

def calculate_overall_score(section_scores):
    weighted_score = sum(
        section_scores[section_name] * weight
        for section_name, weight in SECTION_WEIGHTS.items()
    )

    return float(weighted_score)

resume_df["section_scores"] = resume_df["section_matches"].apply(
    calculate_section_scores
)

resume_df["overall_score"] = resume_df["section_scores"].apply(
    calculate_overall_score
)

score_min = resume_df["overall_score"].min()
score_max = resume_df["overall_score"].max()

if score_max > score_min:
    resume_df["normalized_score"] = (
        (resume_df["overall_score"] - score_min)
        / (score_max - score_min)
    )
else:
    resume_df["normalized_score"] = 1.0

resume_df = resume_df.sort_values(
    "normalized_score",
    ascending=False
).reset_index(drop=True)

print("Ranking completed.")

pprint(
    resume_df[
        [
            "file_name",
            "section_scores",
            "overall_score",
            "normalized_score"
        ]
    ].to_dict("records"),
    width=140
)

import re
import torch
from transformers import pipeline

NER_MODEL_NAME = "yashpwr/resume-ner-bert"

ner_pipeline = pipeline(
    "ner",
    model=NER_MODEL_NAME,
    tokenizer=NER_MODEL_NAME,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN = r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)"
LINKEDIN_PATTERN = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9._%-]+"
GITHUB_PATTERN = r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9._-]+"

def get_header_text(text, max_lines=40, max_chars=1500):
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]
    return "\n".join(lines[:max_lines])[:max_chars]

def extract_contact_information(text):
    emails = re.findall(
        EMAIL_PATTERN,
        text,
        flags=re.IGNORECASE
    )

    phones = []
    for value in re.findall(
        PHONE_PATTERN,
        text
    ):
        digits = re.sub(r"\D", "", value)
        if 10 <= len(digits) <= 13:
            phones.append(value.strip())

    linkedin = re.findall(
        LINKEDIN_PATTERN,
        text,
        flags=re.IGNORECASE
    )

    github = re.findall(
        GITHUB_PATTERN,
        text,
        flags=re.IGNORECASE
    )

    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "linkedin": linkedin[0] if linkedin else None,
        "github": github[0] if github else None
    }

def extract_ner_information(text):
    try:
        entities = ner_pipeline(text)
    except Exception:
        return {}

    extracted = {}

    for entity in entities:
        if entity["score"] < 0.60:
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

        if (
            "NAME" in label_upper
            or label_upper in {"PER", "PERSON"}
        ):
            for value in values:
                words = value.split()
                if 2 <= len(words) <= 4:
                    return value.title()

    header_lines = [
        line.strip()
        for line in header_text.split("\n")
        if line.strip()
    ]

    for line in header_lines[:5]:
        if "@" in line:
            continue

        if re.search(r"\d{4}", line):
            continue

        words = line.split()

        if 2 <= len(words) <= 4 and all(
            re.fullmatch(r"[A-Za-z.'-]+", word)
            for word in words
        ):
            return line.title()

    return None

def extract_candidate_information(text):
    header = get_header_text(text)
    ner_information = extract_ner_information(header)
    contact_information = extract_contact_information(text)

    return {
        "name": extract_name(
            ner_information,
            header
        ),
        "email": contact_information["email"],
        "phone": contact_information["phone"],
        "linkedin": contact_information["linkedin"],
        "github": contact_information["github"],
        "ner_entities": ner_information
    }

resume_df["candidate_information"] = resume_df["text"].apply(
    extract_candidate_information
)

pprint(
    resume_df[
        [
            "file_name",
            "candidate_information"
        ]
    ].to_dict("records"),
    width=160
)

TOP_K = 5

shortlist_df = resume_df.head(TOP_K).copy()

print(f"Shortlisted candidates: {len(shortlist_df)}")

pprint(
    shortlist_df[
        [
            "file_name",
            "candidate_information",
            "section_scores",
            "overall_score",
            "normalized_score"
        ]
    ].to_dict("records"),
    width=160
)

EXPERIENCE_PATTERN = r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b"

def extract_years_of_experience(text):
    matches = re.findall(
        EXPERIENCE_PATTERN,
        text,
        flags=re.IGNORECASE
    )

    if not matches:
        return None

    values = [float(value) for value in matches]

    return max(values)

jd_required_years = extract_years_of_experience(jd_text)

resume_df["years_of_experience"] = resume_df["text"].apply(
    extract_years_of_experience
)

print(f"JD required experience: {jd_required_years}")

pprint(
    resume_df[
        ["file_name", "years_of_experience"]
    ].to_dict("records"),
    width=120
)

def calculate_experience_score(candidate_years, required_years):
    if required_years is None:
        return 1.0

    if candidate_years is None:
        return 0.0

    if candidate_years >= required_years:
        return 1.0

    return float(candidate_years / required_years)

resume_df["experience_numeric_score"] = resume_df[
    "years_of_experience"
].apply(
    lambda value: calculate_experience_score(
        value,
        jd_required_years
    )
)

pprint(
    resume_df[
        [
            "file_name",
            "years_of_experience",
            "experience_numeric_score"
        ]
    ].to_dict("records"),
    width=120
)

SEMANTIC_WEIGHT = 0.90
EXPERIENCE_WEIGHT = 0.10

resume_df["final_score"] = (
    SEMANTIC_WEIGHT * resume_df["overall_score"]
    + EXPERIENCE_WEIGHT * resume_df["experience_numeric_score"]
)

score_min = resume_df["final_score"].min()
score_max = resume_df["final_score"].max()

if score_max > score_min:
    resume_df["normalized_score"] = (
        (resume_df["final_score"] - score_min)
        / (score_max - score_min)
    )
else:
    resume_df["normalized_score"] = 1.0

resume_df = resume_df.sort_values(
    "normalized_score",
    ascending=False
).reset_index(drop=True)

resume_df["rank"] = np.arange(
    1,
    len(resume_df) + 1
)

TOP_K = 5

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1.")

shortlist_df = resume_df.head(TOP_K).copy()

print(f"Shortlist size requested: {TOP_K}")
print(f"Candidates returned: {len(shortlist_df)}")

pprint(
    shortlist_df[
        [
            "rank",
            "file_name",
            "overall_score",
            "experience_numeric_score",
            "final_score",
            "normalized_score"
        ]
    ].to_dict("records"),
    width=140
)

def get_top_evidence(matches, section_name, limit=2):
    evidence = [
        match
        for match in matches
        if match["jd_section"] == section_name
    ]

    return sorted(
        evidence,
        key=lambda item: item["score"],
        reverse=True
    )[:limit]


def generate_match_reason(row):
    matches = row["section_matches"]
    reasons = []

    required_skill_evidence = get_top_evidence(
        matches,
        "required_skills",
        limit=1
    )

    responsibility_evidence = get_top_evidence(
        matches,
        "responsibilities",
        limit=1
    )

    experience_evidence = get_top_evidence(
        matches,
        "experience",
        limit=1
    )

    required_skill_score = row["section_scores"].get(
        "required_skills",
        0.0
    )

    responsibility_score = row["section_scores"].get(
        "responsibilities",
        0.0
    )

    if required_skill_evidence:
        reasons.append(
            f"Strong alignment with required skills "
            f"(match score {required_skill_score:.2f})."
        )

    if responsibility_evidence:
        reasons.append(
            f"Relevant experience and projects align with the role responsibilities "
            f"(match score {responsibility_score:.2f})."
        )

    if row["years_of_experience"] is not None and jd_required_years is not None:
        if row["years_of_experience"] >= jd_required_years:
            reasons.append(
                f"Meets the required experience with "
                f"{row['years_of_experience']:g} years versus "
                f"{jd_required_years:g} required."
            )
        else:
            reasons.append(
                f"Has {row['years_of_experience']:g} years of experience "
                f"versus {jd_required_years:g} required."
            )

    if not reasons and experience_evidence:
        reasons.append(
            "Shows relevant experience aligned with the job description."
        )

    if not reasons:
        reasons.append(
            "Ranked based on semantic similarity between the resume "
            "and job description."
        )

    return " ".join(reasons[:3])


shortlist_df["match_reason"] = shortlist_df.apply(
    generate_match_reason,
    axis=1
)

final_output = []

for _, row in shortlist_df.iterrows():
    candidate = row["candidate_information"]

    final_output.append(
        {
            "rank": int(row["rank"]),
            "name": candidate["name"],
            "email": candidate["email"],
            "phone": candidate["phone"],
            "linkedin": candidate["linkedin"],
            "github": candidate["github"],
            "fit_score": round(
                float(row["normalized_score"]) * 100,
                2
            ),
            "years_of_experience": row["years_of_experience"],
            "reason": row["match_reason"],
            "resume_file": row["file_name"]
        }
    )

print("Final Ranked Shortlist")

pprint(
    final_output,
    width=180
)

import os
import json
import shutil
from pprint import pprint

ARTIFACT_DIR = "/content/resume_screening_artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(f"{ARTIFACT_DIR}/embedding_model", exist_ok=True)
os.makedirs(f"{ARTIFACT_DIR}/ner_model", exist_ok=True)

embedding_model.save(
    f"{ARTIFACT_DIR}/embedding_model/BAAI-bge-large-en-v1.5"
)

ner_model_name = ner_pipeline.model.name_or_path
ner_tokenizer_name = ner_pipeline.tokenizer.name_or_path

ner_pipeline.model.save_pretrained(
    f"{ARTIFACT_DIR}/ner_model"
)

ner_pipeline.tokenizer.save_pretrained(
    f"{ARTIFACT_DIR}/ner_model"
)

config = {
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_device": EMBEDDING_DEVICE,
    "embedding_dimension": embedding_model.get_sentence_embedding_dimension(),
    "normalize_embeddings": True,
    "jd_query_prefix": "Represent this sentence for searching relevant passages: ",
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "ner_model": ner_model_name,
    "ner_tokenizer": ner_tokenizer_name,
    "ner_confidence_threshold": 0.60,
    "semantic_weight": SEMANTIC_WEIGHT,
    "experience_weight": EXPERIENCE_WEIGHT,
    "ranking_uses_personal_details": False
}

with open(f"{ARTIFACT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=4)

preprocessing_config = {
    "lowercase": True,
    "preserve_line_breaks": True,
    "replace_bullets": True,
    "remove_urls": True,
    "allowed_characters": "a-z0-9+#./-,()&@:",
    "collapse_spaces": True,
    "collapse_excess_newlines": True
}

with open(f"{ARTIFACT_DIR}/preprocessing_config.json", "w") as f:
    json.dump(preprocessing_config, f, indent=4)

pprint({
    "artifact_directory": ARTIFACT_DIR,
    "embedding_model": config["embedding_model"],
    "embedding_device": config["embedding_device"],
    "embedding_dimension": config["embedding_dimension"],
    "normalize_embeddings": config["normalize_embeddings"],
    "jd_query_prefix": config["jd_query_prefix"],
    "chunk_size": config["chunk_size"],
    "chunk_overlap": config["chunk_overlap"],
    "ner_model": config["ner_model"],
    "ner_tokenizer": config["ner_tokenizer"],
    "ner_confidence_threshold": config["ner_confidence_threshold"],
    "semantic_weight": config["semantic_weight"],
    "experience_weight": config["experience_weight"],
    "ranking_uses_personal_details": config["ranking_uses_personal_details"]
})

import shutil

zip_path = shutil.make_archive(
    "/content/resume_screening_artifacts",
    "zip",
    "/content/resume_screening_artifacts"
)

print(zip_path)

import os
import json
from pprint import pprint
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

ARTIFACT_DIR = "/content/resume_screening_artifacts"

checks = {}

checks["artifact_directory_exists"] = os.path.isdir(ARTIFACT_DIR)
checks["embedding_model_directory_exists"] = os.path.isdir(
    f"{ARTIFACT_DIR}/embedding_model/BAAI-bge-large-en-v1.5"
)
checks["ner_model_directory_exists"] = os.path.isdir(
    f"{ARTIFACT_DIR}/ner_model"
)
checks["config_exists"] = os.path.isfile(
    f"{ARTIFACT_DIR}/config.json"
)
checks["preprocessing_config_exists"] = os.path.isfile(
    f"{ARTIFACT_DIR}/preprocessing_config.json"
)

with open(f"{ARTIFACT_DIR}/config.json", "r") as f:
    config = json.load(f)

with open(f"{ARTIFACT_DIR}/preprocessing_config.json", "r") as f:
    preprocessing_config = json.load(f)

embedding_model_path = (
    f"{ARTIFACT_DIR}/embedding_model/BAAI-bge-large-en-v1.5"
)

embedding_model = SentenceTransformer(
    embedding_model_path,
    device="cpu"
)

embedding_test = embedding_model.encode(
    ["NLP Engineer with Python and transformer experience"],
    normalize_embeddings=True
)

checks["embedding_model_loads"] = True
checks["embedding_dimension_matches"] = (
    embedding_test.shape[1] == config["embedding_dimension"]
)
checks["embedding_vector_is_finite"] = bool(
    __import__("numpy").isfinite(embedding_test).all()
)

ner_tokenizer = AutoTokenizer.from_pretrained(
    f"{ARTIFACT_DIR}/ner_model"
)

ner_model = AutoModelForTokenClassification.from_pretrained(
    f"{ARTIFACT_DIR}/ner_model"
)

ner_test_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)

ner_test = ner_test_pipeline(
    "Priya Nair is an NLP Engineer with Python experience."
)

checks["ner_model_loads"] = True
checks["ner_inference_runs"] = isinstance(ner_test, list)

pprint({
    "checks": checks,
    "config": config,
    "preprocessing_config": preprocessing_config,
    "embedding_test_shape": embedding_test.shape,
    "ner_test_output": ner_test
})

if all(checks.values()):
    print("\nARTIFACT VERIFICATION: PASS")
else:
    print("\nARTIFACT VERIFICATION: FAIL")
    print("Review the failed checks above.")