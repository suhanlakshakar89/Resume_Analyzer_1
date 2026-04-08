import re
import string
import io
import nltk
import streamlit as st
import numpy as np
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
    "Languages":   ["python","java","c++","c#","javascript","typescript","r","scala","go","rust","kotlin","swift","sql"],
    "ML / AI":     ["machine learning","deep learning","nlp","computer vision","reinforcement learning",
                    "transformers","llm","generative ai","bert","gpt","yolo","opencv","artificial intelligence"],
    "Frameworks":  ["tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm","huggingface",
                    "flask","fastapi","django","streamlit","gradio","react","nodejs","vite","tailwind"],
    "Data":        ["sql","pandas","numpy","spark","hadoop","tableau","power bi","excel",
                    "data analysis","data visualization","etl","dbt","eda","matplotlib","seaborn"],
    "Cloud/MLOps": ["aws","azure","gcp","docker","kubernetes","mlflow","airflow","ci/cd","git","linux","github"],
    "Soft Skills": ["communication","leadership","teamwork","problem solving","agile","scrum","analytical"],
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
    return text

# ── IMPROVED SCORING LOGIC ──
def calculate_match_score(res_text: str, jd_text: str):
    res_clean = clean_text(res_text)
    jd_clean = clean_text(jd_text)
    
    # 1. Skill Match (70% Weight)
    res_skills = set([s for s in ALL_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', res_clean)])
    jd_skills = set([s for s in ALL_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', jd_clean)])
    
    if not jd_skills:
        skill_score = 60.0
    else:
        # Target 70% overlap for a perfect score
        overlap = len(res_skills & jd_skills)
        skill_score = min((overlap / (len(jd_skills) * 0.7)) * 100, 100)

    # 2. Semantic Similarity (30% Weight)
    try:
        # Use char-level matching to catch "Develop" vs "Developer"
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        mat = vec.fit_transform([res_clean, jd_clean])
        semantic_sim = cosine_similarity(mat)[0][1]
        semantic_score = min(semantic_sim * 250, 100) # Normalize to 0-100
    except:
        semantic_score = 0

    final_score = (skill_score * 0.7) + (semantic_score * 0.3)
    return round(min(final_score, 99.0), 1), res_skills, jd_skills

# ── UI ──
st.markdown("""
<style>
    .pill-green { background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:999px; font-size:13px; margin:3px; display:inline-block; border:1px solid #c8e6c9; }
    .pill-red   { background:#fdecea; color:#c62828; padding:4px 12px; border-radius:999px; font-size:13px; margin:3px; display:inline-block; border:1px solid #ffcdd2; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Analyzer")
st.divider()

col1, col2 = st.columns(2, gap="large")
with col1:
    uploaded = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
with col2:
    jd_input = st.text_area("📋 Job Description", height=200, placeholder="Paste the full JD here...")

if st.button("🔍 Analyze Match", use_container_width=True, type="primary"):
    if uploaded and jd_input.strip():
        with st.spinner("⏳ Analyzing..."):
            raw_bytes = uploaded.read()
            ext = uploaded.name.split(".")[-1].lower()
            raw_text = extract_pdf_text(raw_bytes) if ext == "pdf" else extract_docx_text(raw_bytes)
            
            score, res_skills, jd_skills = calculate_match_score(raw_text, jd_input)
            missing = list(jd_skills - res_skills)

            # Display Results
            st.subheader(f"🎯 Match Score: {score}%")
            st.progress(score / 100)
            
            if score >= 75: st.success("✅ Strong match!")
            elif score >= 50: st.warning("⚠️ Moderate match.")
            else: st.error("❌ Weak match.")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🧠 Skills Found")
                if res_skills:
                    pills = "".join([f'<span class="pill-green">{s}</span>' for s in res_skills])
                    st.markdown(pills, unsafe_allow_html=True)
                else: st.write("None detected.")
            
            with col_b:
                st.markdown("### ⚠️ Missing Skills")
                if missing:
                    pills = "".join([f'<span class="pill-red">{s}</span>' for s in missing])
                    st.markdown(pills, unsafe_allow_html=True)
                else: st.write("Perfect coverage!")
    else:
        st.error("Please provide both a resume and a JD.")
