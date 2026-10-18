# Multi-Agent AI Internship & Career Assistant

A multi-agent system that finds internships from several sources, analyzes them with an LLM,
matches them to a student's profile, writes application plans, tracks applications, and
generates a report and notification email. The UI is built with Streamlit.

## Features
- **Data retrieval** from three sources: CSV files, the Remotive jobs API, and a PDF of application guidelines
- **LLM analysis & summarization** (Groq) of each job posting, with retry on rate limits
- **Profile matching** with a rule-based match score and missing-skills detection
- **Automated outputs**: application plans, tracker (CSV), report (Markdown / PDF), email draft (optional SMTP sending)
- **Orchestration**: a Planner agent coordinates the pipeline; LLM calls can run in parallel
- **Evaluation tab**: retrieval hit rate, match statistics, LLM success rate, per-step timing

## Agents
Planner, Job Analysis, Profile Matching, Research, Application, Tracker, Report, Email.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 and paste your Groq API key in the sidebar,
or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GROQ_API_KEY`.
Get a free key at https://console.groq.com/keys.

## Sample data
`sample_data/extra_internships.csv` (fictional companies, for demo only) and
`sample_data/internship_guidelines.pdf` can be uploaded from the sidebar.

## Known limitations
- The match score is keyword-based and does not understand synonyms.
- The tracker lives in the session; download it as CSV to keep it.
- LLM output can contain mistakes and should be reviewed before applying.

## Project structure
```
app.py                       Streamlit app and all agents
requirements.txt
sample_data/                 Demo CSV and guidelines PDF
.streamlit/secrets.toml.example
```
