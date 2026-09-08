from screening import process_screening

jd_text = """
Job Title: Senior Software Engineer
Requirements:
- Strong experience with Python, FastAPI, and Transformers.
- 5+ years of experience in backend development.
"""

resumes = []
for i in range(7):
    resumes.append({
        "file_name": f"resume_{i+1}.txt",
        "text": f"John Doe {i}\nSoftware Engineer with {i+2} years experience.\nSkills: Python, FastAPI, React, Node.js.\nExperience: Built large scale backend systems."
    })

print("Running pipeline...")
results = process_screening(jd_text, resumes, shortlist_size=5)

print(f"Total processed: {results['total_resumes']}")
print(f"Shortlist length: {len(results['results'])}")
for r in results['results']:
    print(f"Rank {r['rank']}: {r['candidate_name']} | Fit Score: {r['fit_score']:.3f} | Normalized: {r.get('normalized_score', 1.0):.3f}")
    if r['evidence']:
        print(f"  Evidence: {len(r['evidence'])} matches found.")
        
print("Validation complete.")
