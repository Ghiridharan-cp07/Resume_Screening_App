# Resume Screening Application — Implementation Specification

You are an expert production web application engineer.

Build a complete, production-quality Resume Screening web application inside the existing project folder:

Resume_Screening_App

The application must use:

Backend:
- Python
- FastAPI

Frontend:
- HTML
- CSS
- JavaScript

Do not use React, Vue, Angular, TypeScript, Tailwind, Bootstrap, or any other frontend framework.

The objective is to turn the already-developed and validated Google Colab resume-screening ML/NLP pipeline into a clean, usable web application.

============================================================
1. CRITICAL RULE — DO NOT BUILD OR TRAIN NEW ML MODELS
============================================================

This is the most important requirement.

The ML/NLP work has already been completed in Google Colab.

A ZIP artifact package is already present inside the project folder.

The ZIP contains the already-created model artifacts and configuration.

You MUST use those existing artifacts.

DO NOT:
- train a new model
- fine-tune a new model
- download and substitute another embedding model
- download and substitute another NER model
- create a replacement model
- use a different embedding model
- use a different tokenizer
- change model architecture
- change the existing model configuration
- introduce another NLP model merely because it seems easier
- recreate the ML pipeline using a different approach

The application is an inference/deployment application only.

No training must happen when the application starts or when a user uploads a document.

The existing artifact package is the source of truth for the deployed models.

============================================================
2. EXISTING ARTIFACTS
============================================================

The project contains a ZIP file containing the model artifacts.

Extract and use the contents of that ZIP as the application's local ML artifacts.

The artifact package contains:

embedding_model/
    BAAI-bge-large-en-v1.5/

ner_model/
    model files
    tokenizer files
    model configuration

config.json

preprocessing_config.json

The embedding model is:

BAAI/bge-large-en-v1.5

The NER model is:

yashpwr/resume-ner-bert

The existing artifact configuration has already been validated.

Important existing configuration:

Embedding dimension:
1024

Chunk size:
350

Chunk overlap:
50

Embeddings:
normalized

JD embedding prefix:
"Represent this sentence for searching relevant passages: "

NER confidence threshold:
0.60

Semantic matching weight:
0.90

Numeric experience weight:
0.10

Personal candidate details:
must NOT be used as ranking signals

Do not replace these values with different values unless the existing artifact configuration explicitly requires it.

The application should load these values from configuration rather than scattering hardcoded paths and settings throughout the code.

============================================================
3. GOOGLE COLAB NOTEBOOK IS THE ML PIPELINE SOURCE OF TRUTH
============================================================

A Google Colab notebook is also provided in the project folder.

Use the notebook as a reference for the exact existing ML/NLP pipeline.

The notebook is NOT a template to redesign.

It is the reference implementation of the already-developed pipeline.

You must carefully inspect the notebook before implementing the backend.

Follow the same conceptual and functional pipeline:

1. Resume/JD text extraction
2. Text preprocessing
3. Resume section splitting
4. JD section splitting
5. Text chunking
6. Embedding generation
7. Section-level semantic comparison
8. Similarity scoring
9. Numeric experience matching
10. Final weighted scoring
11. Ranking
12. Candidate information extraction using the existing NER/regex logic
13. Short explanation generation

Do not introduce additional preprocessing steps simply because they are common in NLP.

Do not add stopword removal if it was not part of the notebook's actual implementation.

Do not stem or lemmatize text if it was not part of the notebook.

Do not change the existing section-detection rules.

Do not change the existing matching strategy.

Do not change the existing scoring strategy.

Do not replace the existing semantic embedding approach with TF-IDF, Word2Vec, FastText, another transformer, or an LLM.

The web application must reproduce the behavior of the validated Colab pipeline as closely as practical.

============================================================
4. ACTUAL EMBEDDING PIPELINE
============================================================

The existing embedding configuration is:

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

EMBEDDING_DEVICE is selected dynamically based on hardware availability.

CHUNK_SIZE = 350

CHUNK_OVERLAP = 50

The application must preserve this configuration.

The existing JD embedding behavior uses:

"Represent this sentence for searching relevant passages: " + JD chunk

The resume embedding behavior must follow the notebook exactly.

Embeddings are normalized before similarity calculation.

Do not change this behavior.

============================================================
5. RESUME INPUT
============================================================

The application must allow the user to upload multiple resumes.

Supported formats:

- PDF
- DOCX
- TXT

Each uploaded resume must be processed independently.

The application must:

1. Identify the file type.
2. Extract text.
3. Apply the existing notebook preprocessing.
4. Split the resume into logical sections using the notebook's section logic.
5. Fall back to full-text processing when the resume is insufficiently structured, following the existing notebook behavior.
6. Chunk relevant sections using the existing chunk configuration.
7. Generate embeddings using the saved BGE model.
8. Perform semantic matching against the selected JD.
9. Extract candidate information.
10. Calculate the final score.
11. Return the ranked result.

Do not require the user to manually enter candidate information.

============================================================
6. JOB DESCRIPTION INPUT
============================================================

The application must allow the user to upload one Job Description.

Supported formats:

- PDF
- DOCX
- TXT

The JD must be processed using the same preprocessing logic as the notebook.

The application must use the notebook's JD section extraction logic.

Important JD sections include concepts such as:

- role
- required skills
- preferred skills
- responsibilities
- experience
- education

The existing notebook's JD heading map and section handling rules must be preserved.

Do not create a completely different JD parser.

============================================================
7. SECTION-BASED SEMANTIC MATCHING
============================================================

Do not compare the entire JD against the entire resume using one single embedding.

Use the section-based strategy developed in the notebook.

Relevant JD sections should be compared against the relevant resume sections according to the notebook's section mapping.

The system should:

1. Chunk JD sections.
2. Chunk resume sections.
3. Generate embeddings.
4. Compare relevant JD chunks with relevant resume chunks.
5. Use cosine similarity through normalized embeddings.
6. Aggregate section-level similarity according to the existing notebook logic.
7. Preserve supporting evidence where available.
8. Produce the semantic matching score.

Do not replace this with simple keyword overlap.

Do not use exact keyword matching as the primary ranking mechanism.

============================================================
8. EXPERIENCE MATCHING
============================================================

The application must preserve the existing numeric experience logic from the notebook.

The system should extract years of experience from the JD and candidate resume.

Use the existing experience extraction pattern and scoring behavior from the notebook.

The final scoring architecture must preserve:

Semantic weight:
0.90

Experience weight:
0.10

Conceptually:

Final Score =
0.90 × Semantic Score
+
0.10 × Experience Score

Do not use candidate name, email, phone number, gender, age, LinkedIn, GitHub, or other personal/contact details as ranking signals.

These fields are extracted for display only.

============================================================
9. CANDIDATE INFORMATION EXTRACTION
============================================================

For each shortlisted candidate, display the candidate information extracted by the existing pipeline.

The application should support displaying:

- Candidate Name
- Email
- Phone
- LinkedIn
- GitHub
- Years of Experience

Use the existing NER model and existing regex/extraction logic from the notebook/artifact configuration.

Do not replace the NER model.

Do not use an LLM for candidate-name extraction.

Do not invent candidate information.

If a field cannot be confidently extracted, display:

Not available

rather than generating or guessing a value.

============================================================
10. FAIRNESS
============================================================

Candidate personal information must never influence ranking.

The ranking must be based on professional relevance, including:

- skills
- experience
- responsibilities
- relevant professional content
- other role-related information already used by the notebook

The following must not be ranking signals:

- name
- gender
- age
- phone number
- email address
- LinkedIn URL
- GitHub URL
- other personal/contact details

These may be displayed after ranking but must remain separate from scoring.

============================================================
11. SHORTLIST SIZE
============================================================

Allow the user to select the number of candidates to return.

Provide a clean shortlist-size control.

Example options:

5
10
15
20

The control should also safely handle cases where fewer resumes are uploaded than the requested shortlist size.

Do not fabricate candidates.

============================================================
12. FINAL RESULT
============================================================

After processing, display candidates in descending order of final fit score.

The main results view should clearly show:

Rank

Candidate Name

Fit Score

Years of Experience

Short Reason

The result should make it immediately obvious which candidate ranked first.

Example conceptual layout:

---------------------------------------------------------
#1  Priya Nair
    Fit Score: 91.4%

    5 years experience

    Strong match in NLP, Python, Transformers and
    semantic responsibilities aligned with the role.
---------------------------------------------------------

#2  Rahul Kumar
    Fit Score: 86.7%

    4 years experience

    Strong Python and NLP alignment with good
    experience overlap.
---------------------------------------------------------

Do not expose unnecessary technical implementation details in the primary result card.

============================================================
13. CANDIDATE DETAIL VIEW
============================================================

Allow the user to expand or open a candidate result to see more information.

Candidate details should include:

Candidate Name
Email
Phone
LinkedIn
GitHub
Years of Experience
Fit Score
Semantic Score
Experience Score
Short Reason

Where available, show relevant supporting evidence from the semantic matching stage.

The interface should make it easy for a recruiter to understand:

Why did this candidate rank highly?

Do not generate long explanations.

The explanation should be concise, professional, and evidence-oriented.

============================================================
14. SCORE DISPLAY
============================================================

The final fit score must be displayed consistently.

Prefer a clear percentage-style presentation:

91.4%

or:

91.4 / 100

Internally preserve the original score values.

Do not manipulate scores merely to make candidates look better.

Do not invent score values.

Clearly distinguish:

Final Fit Score

Semantic Match Score

Experience Score

if all three are displayed.

============================================================
15. UI/UX REQUIREMENTS
============================================================

The frontend must feel like a real premium recruitment/productivity application.

Do NOT create a generic academic ML dashboard.

The UI should be:

- elegant
- modern
- premium
- clean
- professional
- minimal
- responsive
- accessible
- intuitive
- visually polished
- recruiter-friendly

Use a sophisticated color palette with strong contrast and restrained use of accent colors.

Use:

- generous spacing
- elegant typography
- subtle borders
- soft shadows
- polished cards
- clear hierarchy
- refined buttons
- modern upload components
- meaningful hover states
- loading states
- empty states
- error states

Avoid:

- excessive gradients
- excessive animations
- neon colors
- clutter
- unnecessary charts
- unnecessary dashboards
- excessive icons
- excessive rounded elements
- decorative UI that does not improve usability

The interface should feel like a premium SaaS recruitment product.

============================================================
16. MAIN USER FLOW
============================================================

The primary user experience should be extremely simple.

Recommended flow:

Step 1
Upload Job Description

Step 2
Upload multiple resumes

Step 3
Select shortlist size

Step 4
Click:

"Screen Resumes"

Step 5
Show a processing/loading state.

Step 6
Display ranked candidates.

The user should not need to understand the underlying ML pipeline.

============================================================
17. SUGGESTED PAGE STRUCTURE
============================================================

Create one primary application page rather than unnecessarily creating many pages.

Suggested structure:

Header

    Resume Screening
    Intelligent Candidate Matching

Main upload area

    Job Description
    [Upload JD]

    Candidate Resumes
    [Upload Resumes]

    Shortlist Size
    [5 / 10 / 15 / 20]

    [Screen Resumes]

Processing state

    Extracting documents
    Understanding job requirements
    Comparing candidate profiles
    Ranking candidates

Results section

    Screening Summary

    X resumes analyzed
    X candidates shortlisted

    Ranked Candidate Cards

Candidate detail section/modal/drawer

    Candidate information
    Matching scores
    Reason
    Evidence

Keep the entire experience simple.

============================================================
18. BACKEND ARCHITECTURE
============================================================

Use FastAPI.

The backend should be responsible for:

- file upload
- file validation
- text extraction
- preprocessing
- section splitting
- chunking
- model loading
- embedding generation
- NER inference
- semantic matching
- experience scoring
- final ranking
- response serialization

Load the ML models once rather than loading them repeatedly for every candidate.

Do not retrain or reinitialize models for every request.

Use lazy loading or startup loading where appropriate.

Avoid unnecessary repeated model computation.

============================================================
19. MODEL LOADING
============================================================

The application must load the saved local model artifacts.

Do not depend on downloading the model from Hugging Face during every application startup.

Use local artifact paths.

The paths should be centralized through config.py.

Example conceptual configuration:

ARTIFACTS_DIR
EMBEDDING_MODEL_PATH
NER_MODEL_PATH
CONFIG_PATH
PREPROCESSING_CONFIG_PATH

Do not hardcode these paths throughout multiple files.

============================================================
20. CONFIGURATION
============================================================

Create a simple config.py.

config.py should contain centralized application configuration such as:

- project paths
- artifact paths
- upload limits
- supported file extensions
- API configuration
- frontend/static paths
- artifact configuration loading

Where appropriate, read model parameters from the existing config.json rather than duplicating them manually.

The model artifacts remain the source of truth.

============================================================
21. SIMPLE PRODUCTION-ORIENTED PROJECT STRUCTURE
============================================================

Do NOT create a complicated enterprise-style folder hierarchy.

Keep the project intentionally small and maintainable.

Use a structure approximately like:

Resume_Screening_App/
│
├── artifacts/
│   ├── embedding_model/
│   │   └── BAAI-bge-large-en-v1.5/
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
├── requirements.txt
├── README.md
└── resume_screening_prompt.md

If another small file is genuinely necessary, add it only when justified.

Do not create unnecessary layers such as:

- repositories
- factories
- service containers
- multiple configuration modules
- unnecessary utils packages
- unnecessary interfaces
- unnecessary abstractions
- unnecessary microservices

This is one focused application.

Prefer simple, readable modules.

============================================================
22. CODE ORGANIZATION
============================================================

main.py

Responsible primarily for:
- FastAPI application
- routes
- request/response handling
- serving frontend

config.py

Responsible for:
- centralized paths
- configuration
- artifact locations
- application constants

screening.py

Responsible for:
- model loading
- preprocessing
- section splitting
- chunking
- embeddings
- semantic comparison
- experience scoring
- NER extraction
- ranking
- reason generation

templates/index.html

Responsible for:
- application UI structure

static/style.css

Responsible for:
- visual design

static/app.js

Responsible for:
- upload interaction
- API calls
- loading state
- result rendering
- candidate detail interactions
- error handling

requirements.txt

Must contain only dependencies genuinely required by the application.

README.md

Must explain:
- project purpose
- architecture
- how to install dependencies
- how to run FastAPI
- artifact placement
- supported file formats
- application flow
- model/artifact usage
- important configuration
- no-training inference architecture

============================================================
23. DO NOT DUPLICATE THE NOTEBOOK
============================================================

The notebook is a reference implementation.

Do not simply copy the notebook into one giant Python file.

Translate the existing logic into clean application modules while preserving behavior.

The goal is:

Same ML behavior

+

Cleaner production inference code

+

FastAPI backend

+

Premium frontend

Do not change the underlying methodology merely to make the code easier.

============================================================
24. ERROR HANDLING
============================================================

Handle gracefully:

- no JD uploaded
- no resumes uploaded
- unsupported file format
- corrupted PDF
- corrupted DOCX
- empty document
- resume with no extractable text
- JD with no extractable text
- duplicate filenames
- shortlist larger than number of resumes
- model artifact missing
- invalid configuration
- inference failure
- unexpected backend errors

Never expose Python stack traces to the user.

Return clean user-facing error messages.

============================================================
25. FILE HANDLING
============================================================

Do not permanently store uploaded resumes unless explicitly required.

Use temporary processing where possible.

Do not expose uploaded files publicly.

Do not log complete resume contents.

Do not log personal candidate information unnecessarily.

Avoid storing sensitive candidate information in application logs.

============================================================
26. PERFORMANCE
============================================================

The application should avoid unnecessary computation.

Models should be loaded once.

For each screening request:

- load/process the JD once
- generate JD embeddings once
- process each resume once
- generate resume embeddings once
- reuse generated embeddings during comparisons
- perform ranking after scoring

Do not regenerate the same embedding repeatedly.

Do not run training.

Do not run unnecessary model initialization per request.

============================================================
27. API DESIGN
============================================================

Keep the API simple.

Provide a primary screening endpoint.

Conceptually:

POST /api/screen

Input:
- one JD file
- multiple resume files
- shortlist size

Output JSON should contain:

{
    "total_resumes": ...,
    "shortlist_size": ...,
    "results": [
        {
            "rank": ...,
            "candidate_name": ...,
            "email": ...,
            "phone": ...,
            "linkedin": ...,
            "github": ...,
            "years_of_experience": ...,
            "fit_score": ...,
            "semantic_score": ...,
            "experience_score": ...,
            "reason": ...,
            "evidence": [...]
        }
    ]
}

Use appropriate validation and HTTP status codes.

============================================================
28. RESULT ORDER
============================================================

Always sort candidates by final fit score descending.

Highest score must be rank 1.

Ranks must be sequential.

Do not use candidate personal information as a tie-breaker.

If scores are equal, use a deterministic professional-content-based or stable file-order tie-breaker without introducing personal information as a ranking signal.

============================================================
29. REASON GENERATION
============================================================

The reason should be concise.

Example:

"Strong alignment with the required NLP, Python and Transformer skills, supported by relevant experience in NLP application development."

Do not produce generic statements such as:

"This candidate is a good fit."

The reason should reference the actual factors responsible for the ranking.

Do not hallucinate skills or experience that are not present in the resume.

============================================================
30. NO LLM IS REQUIRED FOR THE CORE SCREENING PIPELINE
============================================================

Do not introduce an external generative AI API merely to generate screening scores.

The core screening must remain deterministic and based on the existing pipeline.

Reasons should be generated from the available matching evidence and extracted information.

Do not add OpenAI, Gemini, Claude, or another LLM unless explicitly requested later.

============================================================
31. SECURITY AND PRIVACY
============================================================

Treat resumes as sensitive documents.

Implement:

- file-type validation
- file-size validation
- safe temporary file handling
- no arbitrary file execution
- no public exposure of uploaded files
- no unnecessary persistent candidate storage
- no unnecessary logging of resume contents
- no personal information used for ranking

============================================================
32. RESPONSIVE DESIGN
============================================================

The application must work well on:

- desktop
- laptop
- tablet
- mobile

The primary recruiter workflow should remain easy to use on smaller screens.

============================================================
33. ACCESSIBILITY
============================================================

Use:

- semantic HTML
- accessible labels
- keyboard-friendly controls
- sufficient contrast
- clear focus states
- meaningful error messages
- accessible upload controls

============================================================
34. VISUAL RESULT PRIORITY
============================================================

The most important visual element is the ranked candidate result.

The UI should make the following immediately visible:

1. Candidate name
2. Rank
3. Fit score
4. Years of experience
5. Short reason

Then allow the recruiter to inspect:

- email
- phone
- LinkedIn
- GitHub
- semantic score
- experience score
- evidence

Do not bury the ranking inside a complex dashboard.

============================================================
35. ACCEPTANCE TEST
============================================================

Before considering the application complete, test the complete flow.

Test:

1. Frontend loads.
2. FastAPI starts.
3. Existing embedding model loads successfully.
4. Existing NER model loads successfully.
5. No model training occurs.
6. JD PDF upload works.
7. JD DOCX upload works.
8. JD TXT upload works.
9. Multiple PDF resumes can be uploaded.
10. Multiple DOCX resumes can be uploaded.
11. Multiple TXT resumes can be uploaded.
12. Existing preprocessing is applied.
13. Resume sections are detected correctly.
14. JD sections are detected correctly.
15. Existing chunking configuration is used.
16. BGE embeddings are generated.
17. Embeddings are normalized.
18. Section-level semantic comparison works.
19. Numeric experience matching works.
20. Final weighted score works.
21. Results are sorted descending.
22. Rank 1 is the highest-scoring candidate.
23. Shortlist size works.
24. Candidate information is extracted.
25. Personal details do not affect ranking.
26. A short reason is generated.
27. Evidence is displayed where available.
28. Missing candidate fields are handled gracefully.
29. Empty/invalid files are handled gracefully.
30. Model artifacts are loaded from local files.
31. No model is downloaded/retrained unnecessarily.
32. Models are not reloaded for every candidate.
33. No unnecessary embedding recalculation occurs.
34. API errors are handled cleanly.
35. UI is responsive.
36. Loading states work.
37. Error states work.
38. Final results are visually polished.

============================================================
36. IMPORTANT DEVELOPMENT BEHAVIOR
============================================================

Before writing code:

1. Inspect the provided ZIP artifact.
2. Extract and inspect its directory structure.
3. Inspect config.json.
4. Inspect preprocessing_config.json.
5. Inspect the Google Colab notebook.
6. Understand the existing pipeline.
7. Identify the exact existing model paths.
8. Identify the exact preprocessing behavior.
9. Identify the exact section mappings.
10. Identify the exact matching and scoring logic.
11. Identify the exact NER extraction logic.

Then implement the web application around those existing components.

Do not make assumptions where the notebook already provides the answer.

If there is any conflict between a generic best practice and the actual validated notebook behavior, preserve the notebook behavior unless the change is strictly necessary for safe application integration.

============================================================
37. ABSOLUTE RESTRICTIONS
============================================================

DO NOT:

- train models
- fine-tune models
- create new ML models
- replace BGE
- replace the NER model
- replace the tokenizer
- change chunk size
- change chunk overlap
- change embedding normalization
- change the JD prefix behavior
- add stopword removal unless it exists in the notebook
- add stemming unless it exists in the notebook
- add lemmatization unless it exists in the notebook
- replace semantic matching with keyword matching
- use personal details as ranking signals
- add an LLM for scoring
- introduce unnecessary frameworks
- create a complicated folder structure
- copy the entire notebook into the backend
- create unnecessary databases
- create unnecessary authentication
- create unnecessary user-management functionality
- create unnecessary dashboards
- create unnecessary analytics

The goal is a focused, elegant, production-oriented resume screening application.

============================================================
38. FINAL ENGINEERING PRINCIPLE
============================================================

Think of this project as:

EXISTING ML/NLP PIPELINE
        +
DEPLOYMENT/API LAYER
        +
PREMIUM WEB UI

The ML pipeline already exists.

Your responsibility is to make it usable as a reliable web application.

Do not redesign the machine-learning solution.

Do not retrain anything.

Do not substitute models.

Preserve the validated behavior.

Build only the necessary application infrastructure around it.

The final result should feel like a polished real-world Resume Screening product rather than a demonstration notebook converted into a webpage.