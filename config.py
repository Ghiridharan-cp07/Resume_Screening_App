import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Load Configurations
with open(os.path.join(ARTIFACTS_DIR, "config.json"), "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "preprocessing_config.json"), "r", encoding="utf-8") as f:
    PREPROCESSING_CONFIG = json.load(f)

# Artifact Paths
EMBEDDING_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "embedding_model", "BAAI-bge-large-en-v1.5")
NER_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ner_model")

# File Upload Configuration
SUPPORTED_FORMATS = [".pdf", ".docx", ".txt"]
MAX_UPLOAD_SIZE_MB = 10

import torch

# ML Configuration Constants
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = MODEL_CONFIG.get("chunk_size", 350)
CHUNK_OVERLAP = MODEL_CONFIG.get("chunk_overlap", 50)
JD_QUERY_PREFIX = MODEL_CONFIG.get("jd_query_prefix", "Represent this sentence for searching relevant passages: ")
SEMANTIC_WEIGHT = MODEL_CONFIG.get("semantic_weight", 0.90)
EXPERIENCE_WEIGHT = MODEL_CONFIG.get("experience_weight", 0.10)
NER_CONFIDENCE_THRESHOLD = MODEL_CONFIG.get("ner_confidence_threshold", 0.60)
