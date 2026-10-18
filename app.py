"""
Multi-Agent AI Internship & Career Assistant — Streamlit UI
Run:  streamlit run app.py
"""
import io
import os
import re
import smtplib
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Internship Assistant", page_icon="🤖", layout="wide")

# ───────────────────────────── Data sources ─────────────────────────────
LOCAL_INTERNSHIPS = [
    dict(id=1, company="TechVision AI", role="Computer Vision Intern", location="Cairo, Egypt", type="Hybrid",
         skills="Python, OpenCV, Computer Vision, Machine Learning, NumPy",
         description="Work on computer vision models, image processing and machine learning pipelines.",
         deadline="2026-10-05", source="Local Dataset"),
    dict(id=2, company="Nexa Intelligence", role="AI / LLM Intern", location="Remote", type="Remote",
         skills="Python, LLMs, RAG, LangChain, APIs",
         description="Build LLM applications, retrieval augmented generation systems and AI agents.",
         deadline="2026-10-10", source="Local Dataset"),
    dict(id=3, company="DataCore", role="Data Science Intern", location="New Cairo, Egypt", type="On-site",
         skills="Python, Pandas, SQL, Machine Learning, Statistics",
         description="Analyze business data and develop machine learning models for prediction.",
         deadline="2026-10-15", source="Local Dataset"),
    dict(id=4, company="Smart Systems", role="NLP Intern", location="Giza, Egypt", type="Hybrid",
         skills="Python, NLP, Transformers, HuggingFace, Deep Learning",
         description="Develop NLP applications using transformers and modern language models.",
         deadline="2026-10-20", source="Local Dataset"),
    dict(id=5, company="Future Robotics", role="Robotics Intern", location="New Capital, Egypt", type="On-site",
         skills="C++, ROS, Embedded Systems, Robotics",
         description="Develop robotic systems and embedded control applications.",
         deadline="2026-10-25", source="Local Dataset"),
    dict(id=6, company="CloudAI", role="AI Automation Intern", location="Remote", type="Remote",
         skills="Python, APIs, Automation, LLMs, AI Agents",
         description="Build automated workflows using LLMs, APIs and AI agents.",
         deadline="2026-10-08", source="Local Dataset"),
]

DEFAULT_GUIDELINES = """Internship Application Guidelines.
Students should prioritize internships that match their technical skills and career interests.
AI internships may require Python, machine learning, NLP, computer vision, data analysis, LLMs, RAG, APIs or automation.
Applicants should review the job description carefully before preparing an application.
A strong application should contain a relevant CV, a customized cover letter and evidence of technical projects.
Students should track company name, role, location, deadline and application status.
Remote opportunities may be considered when the student has strong communication skills and can work independently."""

DEFAULT_PROFILE = {
    "name": "AI Student",
    "education": "BSc in Artificial Intelligence",
    "skills": "Python, Machine Learning, Computer Vision, NLP, LLMs, RAG, AI Agents, Data Analysis, Pandas, Scikit-learn, Git, APIs",
    "interests": "Artificial Intelligence, Machine Learning, Computer Vision, LLMs, RAG, AI Agents",
}

STATUSES = ["To Apply", "Preparing", "Applied", "Interview", "Offer", "Rejected"]


def clean_text(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api_jobs(search="artificial intelligence", limit=10):
    try:
        r = requests.get("https://remotive.com/api/remote-jobs", params={"search": search}, timeout=10)
        r.raise_for_status()
        rows = []
        for j in r.json().get("jobs", [])[:limit]:
            rows.append(dict(
                id=f"api_{j.get('id')}", company=j.get("company_name", "Unknown"), role=j.get("title", "Unknown"),
                location=j.get("candidate_required_location", "Remote"), type="Remote", skills="",
                description=j.get("description", ""), deadline="", source="Remotive API"))
        return pd.DataFrame(rows), None
    except Exception as e:  # network / API errors must never crash the app
        return pd.DataFrame(), str(e)


def extract_pdf_text(file_like):
    reader = PdfReader(file_like)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def build_jobs_df(use_api, api_limit, uploaded_csv):
    frames, warnings = [pd.DataFrame(LOCAL_INTERNSHIPS)], []
    if uploaded_csv is not None:
        try:
            frames.append(pd.read_csv(uploaded_csv))
        except Exception as e:
            warnings.append(f"Could not read uploaded CSV: {e}")
    if use_api:
        api_df, err = fetch_api_jobs(limit=api_limit)
        if err:
            warnings.append(f"Remotive API unavailable ({err}) — using local data only.")
        else:
            frames.append(api_df)
    df = pd.concat(frames, ignore_index=True)
    for col in ["id", "company", "role", "location", "type", "skills", "description", "deadline", "source"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["company", "role", "location", "skills", "description"]:
        df[col] = df[col].apply(clean_text)
    df["deadline"] = df["deadline"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["company", "role"]).reset_index(drop=True)
    df["combined_text"] = df["role"] + " " + df["skills"] + " " + df["description"]
    return df, warnings


def build_index(df):
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    return vec, vec.fit_transform(df["combined_text"])


def retrieve_jobs(df, vec, mat, query, top_k):
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    idx = sims.argsort()[::-1][:top_k]
    out = df.iloc[idx].copy()
    out["retrieval_score"] = sims[idx]
    return out.reset_index(drop=True)


# ───────────────────────────── LLM helper ─────────────────────────────
def _retry_wait(msg):
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", msg)
    if m:
        return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 40)
    return 15


def call_llm(client, prompt, model, max_tokens=900, retries=4):
    if client is None:
        return "LLM unavailable — add a Groq API key in the sidebar."
    last = ""
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.2, max_tokens=max_tokens,
                messages=[{"role": "system", "content": "You are a professional AI assistant. Provide accurate, concise and structured answers."},
                          {"role": "user", "content": prompt}])
            return r.choices[0].message.content
        except Exception as e:
            last = str(e)
            if ("429" in last or "rate_limit" in last.lower()) and attempt < retries - 1:
                time.sleep(_retry_wait(last))  # wait for the rate-limit window, then retry
                continue
            break
    return f"LLM Error: {last}"


# ───────────────────────────── Agents ─────────────────────────────
class BaseAgent:
    def __init__(self, name, role, logs):
        self.name, self.role, self.logs = name, role, logs

    def log(self, msg):
        self.logs.append(f"{datetime.now():%H:%M:%S}  [{self.name}] {msg}")


class PlannerAgent(BaseAgent):
    def create_plan(self, request):
        self.log("Creating workflow plan.")
        return ["Retrieve internship opportunities", "Analyze job requirements", "Match jobs with student profile",
                "Research application guidelines", "Generate personalized application plans",
                "Update application tracker", "Generate final report & email"]


class JobAnalysisAgent(BaseAgent):
    def __init__(self, logs, client, model):
        super().__init__("Job Analysis Agent", "Extracts requirements from listings", logs)
        self.client, self.model = client, model

    def analyze(self, job):
        self.log(f"Analyzing: {job['role']} @ {job['company']}")
        prompt = f"""Analyze this internship.
Company: {job['company']}
Role: {job['role']}
Location: {job['location']}
Skills: {job['skills']}
Description: {job['description'][:1500]}

Extract: 1) Main responsibilities 2) Required technical skills 3) Preferred skills
4) Key technologies 5) Candidate profile 6) Important application considerations.
Return concise structured bullet points (max 180 words)."""
        return call_llm(self.client, prompt, self.model, max_tokens=450)


class ProfileMatchingAgent(BaseAgent):
    WATCHLIST = ["SQL", "Docker", "AWS", "TensorFlow", "PyTorch", "LangChain", "Statistics"]

    def __init__(self, logs):
        super().__init__("Profile Matching Agent", "Scores fit with the student's profile", logs)

    @staticmethod
    def _has(term, text):  # whole-word match so "rag" doesn't match "storage"
        return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text) is not None

    def match(self, job, skills):
        self.log(f"Matching: {job['role']} @ {job['company']}")
        text = f"{job['skills']} {job['description']} {job['role']}".lower()
        mine = {s.lower() for s in skills}
        matched = sorted(s for s in mine if self._has(s, text))
        missing = [s for s in self.WATCHLIST if self._has(s, text) and s.lower() not in mine]
        return {"matched_skills": matched, "missing_skills": missing,
                "match_score": round(len(matched) / max(len(mine), 1) * 100, 2)}


class ResearchAgent(BaseAgent):
    def __init__(self, text, logs):
        super().__init__("Research Agent", "Searches reference documents", logs)
        self.text = text

    def research(self, question, k=5):
        self.log("Searching reference document.")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", self.text) if s.strip()]
        kws = set(re.findall(r"\w+", question.lower()))
        scored = [(sum(w in s.lower() for w in kws), s) for s in sentences]
        scored = sorted([x for x in scored if x[0] > 0], key=lambda x: -x[0])
        return [s for _, s in scored[:k]]


class ApplicationAgent(BaseAgent):
    def __init__(self, logs, client, model):
        super().__init__("Application Agent", "Writes application plans", logs)
        self.client, self.model = client, model

    def create_plan(self, job, match, analysis, profile, guidelines):
        self.log(f"Preparing plan for {job['company']}")
        prompt = f"""Create a personalized internship application plan.
Student: {profile['education']}
Skills: {profile['skills']}
Interests: {profile['interests']}
Company: {job['company']} | Role: {job['role']} | Location: {job['location']}
Match score: {match['match_score']}% | Matched: {match['matched_skills']} | Missing: {match['missing_skills']}
Job analysis: {str(analysis)[:1200]}
Guidelines to respect: {guidelines}

Create: 1) Why this role matches 2) Skills to highlight in the CV 3) Projects to highlight
4) Missing skills to review 5) Cover-letter points 6) Application checklist.
Do not invent experience. Keep it under 300 words."""
        return call_llm(self.client, prompt, self.model, max_tokens=700)


class TrackerAgent(BaseAgent):
    def __init__(self, logs):
        super().__init__("Tracker Agent", "Maintains the application tracker", logs)

    def create_tracker(self, ranked):
        self.log(f"Tracking {len(ranked)} opportunities.")
        t = ranked[["company", "role", "location", "match_score", "deadline"]].copy()
        t["status"] = "To Apply"
        t["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return t.reset_index(drop=True)


class ReportAgent(BaseAgent):
    def __init__(self, logs):
        super().__init__("Report Agent", "Generates the final report", logs)

    def generate(self, tracker, plans, guidelines_hits):
        self.log("Generating report.")
        out = ["# AI Internship Research Report\n", f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n",
               f"Total opportunities analyzed: {len(tracker)}\n", "\n## Recommended Opportunities\n"]
        for _, r in tracker.iterrows():
            out.append(f"\n### {r['company']} — {r['role']}\n\n- Location: {r['location']}\n"
                       f"- Match Score: {r['match_score']}%\n- Deadline: {r['deadline'] or '—'}\n- Status: {r['status']}\n")
        if guidelines_hits:
            out.append("\n## Application Guidelines Applied\n")
            out.extend(f"- {g}" for g in guidelines_hits)
        out.append("\n## Application Plans\n")
        out.extend(f"\n{p}\n" for p in plans)
        return "\n".join(out)


class EmailAgent(BaseAgent):
    def __init__(self, logs):
        super().__init__("Email Agent", "Drafts notification emails", logs)

    def create_email(self, tracker):
        self.log("Drafting notification email.")
        top = tracker.sort_values("match_score", ascending=False).head(5)
        body = "Hello,\n\nThe AI Internship Assistant found the following opportunities:\n\n"
        for _, r in top.iterrows():
            body += f"- {r['company']} | {r['role']} | Match: {r['match_score']}% | Deadline: {r['deadline'] or '—'}\n"
        body += "\nPlease review the application plans and apply before the deadlines.\n\nGenerated automatically by the Multi-Agent AI System."
        return {"subject": "AI Internship Assistant - Recommended Opportunities", "body": body}


# ───────────────────────────── Exports ─────────────────────────────
def report_to_pdf(md_text):
    buf = io.BytesIO()
    doc, styles, story = SimpleDocTemplate(buf, pagesize=A4), getSampleStyleSheet(), []
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for line in md_text.split("\n"):
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(esc(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(esc(line[4:]), styles["Heading3"]))
        elif line.strip():
            story.append(Paragraph(esc(line.replace("**", "")), styles["BodyText"]))
            story.append(Spacer(1, 4))
    doc.build(story)
    return buf.getvalue()


def send_email(host, port, user, password, to, subject, body):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(host, int(port), context=ssl.create_default_context(), timeout=20) as s:
        s.login(user, password)
        s.send_message(msg)


# ───────────────────────────── Pipeline ─────────────────────────────
def run_pipeline(cfg, status):
    logs, timings, t_all = [], {}, time.time()
    client = Groq(api_key=cfg["api_key"]) if cfg["api_key"] else None
    profile = cfg["profile"]
    skills = [s.strip() for s in profile["skills"].split(",") if s.strip()]

    planner = PlannerAgent("Planner Agent", "Coordinates", logs)
    analyzer = JobAnalysisAgent(logs, client, cfg["model"])
    matcher = ProfileMatchingAgent(logs)
    researcher = ResearchAgent(cfg["guidelines"], logs)
    applicant = ApplicationAgent(logs, client, cfg["model"])
    tracker_agent, reporter, emailer = TrackerAgent(logs), ReportAgent(logs), EmailAgent(logs)

    def step(label, fn):
        status.write(label)
        t = time.time()
        res = fn()
        timings[label] = round(time.time() - t, 2)
        return res

    plan = step("🧭 Planner Agent — building workflow", lambda: planner.create_plan(cfg["request"]))

    def retrieve():
        df, warns = build_jobs_df(cfg["use_api"], cfg["api_limit"], cfg["csv"])
        vec, mat = build_index(df)
        query = f"{profile['skills']} {profile['interests']}"
        return df, warns, vec, mat, retrieve_jobs(df, vec, mat, query, cfg["top_k"])

    df, warns, vec, mat, retrieved = step("📥 Retrieval — CSV + API + TF-IDF search", retrieve)

    jobs = [row for _, row in retrieved.iterrows()]
    analyses = step(f"🔍 Job Analysis Agent — {len(jobs)} listings in parallel",
                    lambda: list(ThreadPoolExecutor(max_workers=cfg['workers']).map(analyzer.analyze, jobs)))
    matches = step("🎯 Profile Matching Agent", lambda: [matcher.match(j, skills) for j in jobs])
    hits = step("📚 Research Agent — guidelines lookup",
                lambda: researcher.research("internship application CV cover letter skills deadline"))

    ranked = pd.DataFrame([{**j[["company", "role", "location", "skills", "description", "deadline", "source",
                                  "retrieval_score"]].to_dict(),
                            **m, "analysis": a} for j, m, a in zip(jobs, matches, analyses)])
    ranked = ranked.sort_values("match_score", ascending=False).reset_index(drop=True)

    def plans_fn():
        top = ranked.head(cfg["n_plans"])
        def one(row):
            m = {k: row[k] for k in ("match_score", "matched_skills", "missing_skills")}
            p = applicant.create_plan(row, m, row["analysis"], profile, " ".join(hits))
            return f"## {row['company']} — {row['role']}\n\nMatch Score: {row['match_score']}%\n\n{p}\n"
        return list(ThreadPoolExecutor(max_workers=cfg['workers']).map(one, [r for _, r in top.iterrows()]))

    plans = step(f"✍️ Application Agent — top {cfg['n_plans']} plans", plans_fn)
    tracker = step("🗂️ Tracker Agent", lambda: tracker_agent.create_tracker(ranked))
    report = step("📄 Report Agent", lambda: reporter.generate(tracker, plans, hits))
    email = step("✉️ Email Agent", lambda: emailer.create_email(tracker))

    timings["TOTAL"] = round(time.time() - t_all, 2)
    return dict(plan=plan, df=df, warnings=warns, vec=vec, mat=mat, retrieved=retrieved, ranked=ranked, hits=hits,
                plans=plans, tracker=tracker, report=report, email=email, logs=logs, timings=timings,
                llm_err=next((str(a) for a in analyses if str(a).startswith(("LLM Error", "LLM unavailable"))), ""),
                llm_ok=sum(not str(a).startswith(("LLM Error", "LLM unavailable")) for a in analyses),
                llm_total=len(analyses), generated=datetime.now().strftime("%Y-%m-%d %H:%M"))


# ───────────────────────────── Sidebar ─────────────────────────────
def get_default_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


@st.cache_data(ttl=600, show_spinner=False)
def list_groq_models(key):
    """Ask Groq which chat models this key can actually use."""
    try:
        ids = [m.id for m in Groq(api_key=key).models.list().data]
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "distil")
        ids = sorted(i for i in ids if not any(x in i.lower() for x in skip))
        # put the larger general-purpose models first
        ids.sort(key=lambda i: (0 if "70b" in i or "120b" in i else 1, i))
        return ids, None
    except Exception as e:
        return [], str(e)


with st.sidebar:
    st.title("🤖 Control Panel")
    api_key = st.text_input("Groq API key", value=get_default_key(), type="password",
                            help="Stored only in this session. Or set GROQ_API_KEY in .streamlit/secrets.toml")
    models, model_err = list_groq_models(api_key) if api_key else ([], None)
    if api_key and models:
        model = st.selectbox("LLM model (available on your key)", models)
    else:
        model = st.selectbox("LLM model", FALLBACK_MODELS)
        if api_key and model_err:
            st.caption(f"Could not fetch model list: {model_err[:120]}")

    st.subheader("👩‍🎓 Student profile")
    prof = {k: st.text_area(k.title(), v, height=90 if k in ("skills", "interests") else 40) if k in ("skills", "interests")
            else st.text_input(k.title(), v) for k, v in DEFAULT_PROFILE.items()}

    st.subheader("📦 Data sources")
    use_api = st.toggle("Live jobs (Remotive API)", value=True)
    api_limit = st.slider("API jobs to fetch", 3, 20, 10, disabled=not use_api)
    up_csv = st.file_uploader("Extra internships CSV (optional)", type="csv")
    up_pdf = st.file_uploader("Guidelines PDF (optional)", type="pdf")

    st.subheader("⚙️ Pipeline")
    top_k = st.slider("Jobs to retrieve & analyze", 3, 12, 6)
    n_plans = st.slider("Application plans to write", 1, 5, 3)
    workers = st.slider("Parallel LLM calls", 1, 3, 1,
                        help="Use 1 on Groq's free tier to avoid rate limits; raise it if you have a paid tier.")
    request = st.text_area("Your request", "Find the most suitable AI internships for me, analyze them, match them with my skills, "
                           "prepare application plans, track them and generate a final report.", height=100)
    run = st.button("🚀 Run multi-agent pipeline", type="primary", width="stretch")

# ───────────────────────────── Main ─────────────────────────────
st.title("Multi-Agent AI Internship & Career Assistant")
st.caption("Planner → Retrieval → Job Analysis → Profile Matching → Research → Application → Tracker → Report → Email")

if run:
    guidelines = DEFAULT_GUIDELINES
    if up_pdf is not None:
        try:
            guidelines = extract_pdf_text(up_pdf) or DEFAULT_GUIDELINES
        except Exception as e:
            st.warning(f"Could not read the PDF ({e}); using default guidelines.")
    cfg = dict(api_key=api_key, model=model, profile=prof, use_api=use_api, api_limit=api_limit, csv=up_csv,
               guidelines=guidelines, top_k=min(top_k, 50), n_plans=min(n_plans, top_k), request=request, workers=workers)
    with st.status("Running multi-agent pipeline…", expanded=True) as status:
        try:
            st.session_state["run"] = run_pipeline(cfg, status)
            st.session_state["tracker_edit"] = st.session_state["run"]["tracker"].copy()
            status.update(label="Pipeline complete ✅", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline failed ❌", state="error")
            st.exception(e)

R = st.session_state.get("run")
if not R:
    st.info("Set your profile in the sidebar and press **Run multi-agent pipeline**.")
    st.stop()

for w in R["warnings"]:
    st.warning(w)
if R["llm_ok"] < R["llm_total"]:
    st.warning("Some LLM calls failed — analyses/plans may contain error messages.")
    st.error(f"Actual error from the LLM: {R['llm_err']}")

tab_over, tab_jobs, tab_plans, tab_track, tab_report, tab_eval, tab_logs = st.tabs(
    ["📊 Overview", "🔎 Opportunities", "📝 Application Plans", "✅ Tracker", "📄 Report & Email", "🧪 Evaluation", "🪵 Agent Logs"])

with tab_over:
    ranked = R["ranked"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs in database", len(R["df"]))
    c2.metric("Analyzed", len(ranked))
    c3.metric("Best match", f"{ranked['match_score'].max():.0f}%")
    c4.metric("Pipeline time", f"{R['timings']['TOTAL']}s")
    st.subheader("Workflow plan")
    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(R["plan"], 1)))
    st.subheader("Match score by internship")
    chart = ranked.assign(label=ranked["company"] + " — " + ranked["role"]).set_index("label")["match_score"]
    st.bar_chart(chart, horizontal=True)

with tab_jobs:
    st.dataframe(R["ranked"][["company", "role", "location", "match_score", "retrieval_score", "deadline", "source"]],
                 width="stretch", hide_index=True,
                 column_config={"match_score": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.0f"),
                                "retrieval_score": st.column_config.NumberColumn("Retrieval", format="%.3f")})
    for _, r in R["ranked"].iterrows():
        with st.expander(f"{r['company']} — {r['role']}  ({r['match_score']}%)"):
            a, b = st.columns(2)
            a.markdown("**Matched skills**\n\n" + (", ".join(r["matched_skills"]) or "—"))
            b.markdown("**Missing skills**\n\n" + (", ".join(r["missing_skills"]) or "—"))
            st.markdown("**LLM analysis**")
            st.markdown(r["analysis"])

with tab_plans:
    for p in R["plans"]:
        with st.expander(p.split("\n")[0].replace("## ", ""), expanded=False):
            st.markdown(p)

with tab_track:
    st.caption("Edit the status directly — changes feed the downloadable CSV.")
    edited = st.data_editor(
        st.session_state["tracker_edit"], width="stretch", hide_index=True,
        disabled=["company", "role", "location", "match_score", "deadline", "created_at"],
        column_config={"status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True)},
        key="tracker_editor")
    st.session_state["tracker_edit"] = edited
    st.download_button("⬇️ Download tracker CSV", edited.to_csv(index=False).encode("utf-8"),
                       "internship_tracker.csv", "text/csv")
    st.bar_chart(edited["status"].value_counts())

with tab_report:
    st.markdown(R["report"])
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Markdown", R["report"], "internship_report.md", "text/markdown", width="stretch")
    d2.download_button("⬇️ PDF", report_to_pdf(R["report"]), "internship_report.pdf", "application/pdf",
                       width="stretch")
    st.divider()
    st.subheader("✉️ Notification email")
    subj = st.text_input("Subject", R["email"]["subject"])
    body = st.text_area("Body", R["email"]["body"], height=220)
    with st.expander("Send via SMTP (Gmail → use an App Password)"):
        s1, s2 = st.columns(2)
        host = s1.text_input("SMTP host", "smtp.gmail.com")
        port = s2.text_input("SMTP SSL port", "465")
        user = s1.text_input("Your email")
        pwd = s2.text_input("App password", type="password")
        to = st.text_input("Send to", user)
        if st.button("Send email"):
            try:
                send_email(host, port, user, pwd, to, subj, body)
                st.success(f"Email sent to {to}")
            except Exception as e:
                st.error(f"Failed to send: {e}")

with tab_eval:
    st.subheader("Retrieval quality")
    cases = {"Python machine learning internship": "machine learning|data science", "LLM RAG AI agent internship": "llm|agent|automation",
             "computer vision internship": "computer vision", "NLP transformer internship": "nlp",
             "robotics embedded ROS": "robotics"}
    rows = []
    for q, expected in cases.items():
        top = retrieve_jobs(R["df"], R["vec"], R["mat"], q, 3)
        hit = bool(re.search(expected, (top.iloc[0]["role"] + " " + top.iloc[0]["skills"]).lower()))
        rows.append({"query": q, "top result": f"{top.iloc[0]['company']} — {top.iloc[0]['role']}", "relevant@1": hit})
    ev = pd.DataFrame(rows)
    st.dataframe(ev, width="stretch", hide_index=True)
    st.metric("Hit rate @1", f"{ev['relevant@1'].mean() * 100:.0f}%")

    st.subheader("Matching & reliability")
    s = R["ranked"]["match_score"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg match", f"{s.mean():.1f}%")
    m2.metric("Max", f"{s.max():.1f}%")
    m3.metric("Min", f"{s.min():.1f}%")
    m4.metric("LLM success", f"{R['llm_ok']}/{R['llm_total']}")
    st.subheader("Execution time per step (s)")
    st.bar_chart(pd.Series({k: v for k, v in R["timings"].items() if k != "TOTAL"}), horizontal=True)

with tab_logs:
    st.code("\n".join(R["logs"]), language="text")
