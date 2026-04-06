import re
import string
import io
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from docx import Document

# ── page config ──
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="centered")

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

# ── skills database ──
SKILLS_DB = {
    "Languages":   ["python","java","c++","c#","javascript","typescript","r","scala","go","rust","kotlin","swift"],
    "ML / AI":     ["machine learning","deep learning","nlp","computer vision","reinforcement learning",
                    "transformers","llm","generative ai","bert","gpt","yolo","opencv"],
    "Frameworks":  ["tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm","huggingface",
                    "flask","fastapi","django","streamlit","gradio","react","nodejs"],
    "Data":        ["sql","pandas","numpy","spark","hadoop","tableau","power bi","excel",
                    "data analysis","data visualization","etl","dbt"],
    "Cloud/MLOps": ["aws","azure","gcp","docker","kubernetes","mlflow","airflow","ci/cd","git","linux"],
    "Soft Skills": ["communication","leadership","teamwork","problem solving","agile","scrum"],
}
ALL_SKILLS = [s for grp in SKILLS_DB.values() for s in grp]

# ── helpers ──
def extract_pdf_text(file_bytes: bytes) -> str:
    output = io.StringIO()
    extract_text_to_fp(io.BytesIO(file_bytes), output, laparams=LAParams())
    return output.getvalue()

def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join([w for w in text.split() if w not in STOP_WORDS and len(w) > 1])

def calculate_similarity(res: str, jd: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform([res, jd])
    return round(cosine_similarity(mat)[0][1] * 100, 2)

def extract_skills(text: str) -> dict:
    return {cat: [s for s in skills if s in text]
            for cat, skills in SKILLS_DB.items()
            if any(s in text for s in skills)}

def get_missing(res_text: str, jd_text: str) -> list:
    return list(set(s for s in ALL_SKILLS if s in jd_text)
              - set(s for s in ALL_SKILLS if s in res_text))

# ── UI ──
st.markdown("""
<style>
    .pill-green { background:#e8f5e9; color:#2e7d32; padding:4px 12px;
                  border-radius:999px; font-size:13px; margin:3px; display:inline-block; }
    .pill-red   { background:#fdecea; color:#c62828; padding:4px 12px;
                  border-radius:999px; font-size:13px; margin:3px; display:inline-block; }
    .score-bar-bg { background:#f0f0f0; border-radius:999px; height:14px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Analyzer")
st.markdown("Upload your resume and paste a job description — get an instant match score, skill gap analysis, and actionable feedback.")
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Resume")
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

with col2:
    st.subheader("📋 Job Description")
    jd_input = st.text_area("Paste JD here", height=220, placeholder="Copy and paste the full job description...")

st.divider()
run = st.button("🔍 Analyze Now", use_container_width=True, type="primary")

if run:
    if not uploaded:
        st.error("❌ Please upload your resume.")
        st.stop()
    if not jd_input.strip():
        st.error("❌ Please paste a job description.")
        st.stop()

    with st.spinner("⏳ Analyzing your resume..."):
        raw_bytes = uploaded.read()
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        raw_text = extract_pdf_text(raw_bytes) if ext == "pdf" else extract_docx_text(raw_bytes)

        if not raw_text.strip():
            st.error("❌ Could not extract text. Is your PDF a scanned image?")
            st.stop()

        res_clean = clean_text(raw_text)
        jd_clean  = clean_text(jd_input)
        score     = calculate_similarity(res_clean, jd_clean)
        res_skills = extract_skills(res_clean)
        missing    = get_missing(res_clean, jd_clean)

    # ── score ──
    st.subheader("🎯 Match Score")
    if score >= 75:
        color, label, icon = "green",  "Strong match — ready to apply!",  "✅"
    elif score >= 50:
        color, label, icon = "orange", "Moderate match — some gaps found.", "⚠️"
    else:
        color, label, icon = "red",    "Weak match — significant gaps.",   "❌"

    c1, c2, c3 = st.columns([2, 1, 2])
    c1.metric("Score", f"{score}%")
    c2.markdown(f"<div style='font-size:32px;text-align:center;padding-top:8px'>{icon}</div>", unsafe_allow_html=True)
    c3.metric("Verdict", label[:20] + "...")

    st.progress(int(score) / 100)
    st.markdown(f"**{icon} {label}**")

    if score < 75 and missing:
        st.info(f"💡 Top skills to add: **{', '.join(missing[:5])}**")

    st.divider()

    # ── skills found ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🧠 Skills Found in Resume")
        if res_skills:
            for cat, skills in res_skills.items():
                st.markdown(f"**{cat}**")
                pills = " ".join(f'<span class="pill-green">{s}</span>' for s in skills)
                st.markdown(pills, unsafe_allow_html=True)
        else:
            st.warning("No known skills detected. Try expanding the skills database.")

    with col_b:
        st.subheader("⚠️ Missing Skills (from JD)")
        if missing:
            pills = " ".join(f'<span class="pill-red">{s}</span>' for s in missing)
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.success("🎉 No missing skills — great coverage!")

    st.divider()

    # ── raw text preview ──
    with st.expander("🔍 View extracted resume text"):
        st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))
