document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('screening-form');
    const jdFileInput = document.getElementById('jd_file');
    const resumeFilesInput = document.getElementById('resume_files');
    const jdFileName = document.getElementById('jd-file-name');
    const resumeFilesList = document.getElementById('resume-files-list');
    
    const processingState = document.getElementById('processing-state');
    const resultsSection = document.getElementById('results-section');
    const candidateCards = document.getElementById('candidate-cards');
    const summaryTotal = document.getElementById('total-analyzed');
    const summaryShortlist = document.getElementById('total-shortlisted');

    const modal = document.getElementById('candidate-modal');
    const closeModal = document.querySelector('.close-modal');

    // Current Candidates Data
    let currentCandidates = [];

    // File Input UI Updates
    jdFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            jdFileName.textContent = e.target.files[0].name;
            jdFileName.style.color = 'var(--primary-color)';
        } else {
            jdFileName.textContent = '';
        }
    });

    resumeFilesInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const count = e.target.files.length;
            resumeFilesList.textContent = `${count} file${count > 1 ? 's' : ''} selected`;
            resumeFilesList.style.color = 'var(--primary-color)';
        } else {
            resumeFilesList.textContent = '';
        }
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (jdFileInput.files.length === 0 || resumeFilesInput.files.length === 0) {
            alert('Please upload both Job Description and Resumes.');
            return;
        }

        const formData = new FormData();
        formData.append('jd_file', jdFileInput.files[0]);
        for (let i = 0; i < resumeFilesInput.files.length; i++) {
            formData.append('resume_files', resumeFilesInput.files[i]);
        }
        formData.append('shortlist_size', document.getElementById('shortlist_size').value);

        // Show processing state
        processingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        window.scrollTo({ top: processingState.offsetTop, behavior: 'smooth' });

        try {
            const response = await fetch('/api/screen', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process resumes');
            }

            const data = await response.json();
            currentCandidates = data.results;
            
            renderResults(data);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            processingState.classList.add('hidden');
        }
    });

    function renderResults(data) {
        summaryTotal.textContent = data.total_resumes;
        summaryShortlist.textContent = data.results.length;
        candidateCards.innerHTML = '';

        data.results.forEach((candidate, index) => {
            const card = document.createElement('div');
            card.className = 'candidate-card';
            card.dataset.index = index;

            const rank = candidate.rank;
            const scoreDisplay = candidate.display_score ? `${candidate.display_score}` : `${(candidate.fit_score * 100).toFixed(1)}`;
            const expText = candidate.years_of_experience !== null ? `${candidate.years_of_experience} years experience` : 'Experience not specified';
            
            card.innerHTML = `
                <div class="card-rank">#${rank}</div>
                <div class="card-content">
                    <div class="card-header-row">
                        <div class="card-name">${candidate.candidate_name}</div>
                        <div class="card-score">${scoreDisplay}</div>
                    </div>
                    <div class="card-meta">${expText}</div>
                    <div class="card-reason">${candidate.reason}</div>
                </div>
            `;

            card.addEventListener('click', () => openModal(index));
            candidateCards.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
        window.scrollTo({ top: resultsSection.offsetTop - 40, behavior: 'smooth' });
    }

    // Modal Logic
    function openModal(index) {
        const candidate = currentCandidates[index];
        if (!candidate) return;

        document.getElementById('modal-rank').textContent = `#${candidate.rank}`;
        document.getElementById('modal-name').textContent = candidate.candidate_name;
        
        const scoreDisplay = candidate.display_score ? `${candidate.display_score}` : `${(candidate.fit_score * 100).toFixed(1)}`;
        document.getElementById('modal-fit-score').textContent = scoreDisplay;

        // Info Grid
        document.getElementById('modal-email').textContent = candidate.email || 'Not available';
        document.getElementById('modal-phone').textContent = candidate.phone || 'Not available';
        
        const linkedinEl = document.getElementById('modal-linkedin');
        if (candidate.linkedin) {
            linkedinEl.textContent = 'View Profile';
            linkedinEl.href = candidate.linkedin.startsWith('http') ? candidate.linkedin : `https://${candidate.linkedin}`;
        } else {
            linkedinEl.textContent = 'Not available';
            linkedinEl.removeAttribute('href');
        }

        const githubEl = document.getElementById('modal-github');
        if (candidate.github) {
            githubEl.textContent = 'View Profile';
            githubEl.href = candidate.github.startsWith('http') ? candidate.github : `https://${candidate.github}`;
        } else {
            githubEl.textContent = 'Not available';
            githubEl.removeAttribute('href');
        }

        document.getElementById('modal-experience').textContent = candidate.years_of_experience !== null ? `${candidate.years_of_experience} years` : 'Not available';

        // Scores
        const semScorePct = (candidate.semantic_score * 100).toFixed(1);
        const expScorePct = (candidate.experience_score * 100).toFixed(1);

        document.getElementById('modal-semantic-val').textContent = `${semScorePct}%`;
        document.getElementById('modal-semantic-fill').style.width = `${semScorePct}%`;

        document.getElementById('modal-experience-val').textContent = `${expScorePct}%`;
        document.getElementById('modal-experience-fill').style.width = `${expScorePct}%`;

        // Reasoning & Evidence
        document.getElementById('modal-reason').textContent = candidate.reason;

        const evidenceList = document.getElementById('modal-evidence-list');
        evidenceList.innerHTML = '';
        if (candidate.evidence && candidate.evidence.length > 0) {
            candidate.evidence.forEach(ev => {
                const li = document.createElement('li');
                li.className = 'evidence-item';
                li.innerHTML = `
                    <div class="evidence-header">
                        <span class="badge-jd">${ev.jd_section.replace('_', ' ')}</span>
                        <span class="evidence-arrow">→</span>
                        <span class="badge-resume">${ev.resume_section.replace('_', ' ')}</span>
                        <span class="evidence-score">${(ev.score * 100).toFixed(1)}% Match</span>
                    </div>
                    <div class="evidence-text">"... ${ev.match_text.trim()} ..."</div>
                `;
                evidenceList.appendChild(li);
            });
            document.getElementById('modal-evidence-container').classList.remove('hidden');
        } else {
            document.getElementById('modal-evidence-container').classList.add('hidden');
        }

        modal.classList.remove('hidden');
    }

    closeModal.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});
