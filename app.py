"""
Study Buddy AI — Complete Professional Rebuild
A Gemini-powered AI study assistant with full document understanding,
interactive quiz grading, animated flashcards, and smart revision planning.
"""

import streamlit as st
import os
import json
import base64
from pathlib import Path

# ── Page config first ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Study Buddy AI — Powered by Google Gemini"}
)

from config import APP_NAME, GEMINI_API_KEY, OPENAI_API_KEY, SAMPLE_DATA_DIR, ASSETS_DIR
from agent import create_study_buddy_agent
from utils.pdf_utils import extract_text_from_pdf_stream
from utils.text_cleaner import clean_text, truncate_text
from utils.validators import validate_file_size
from utils.helper import shuffle_flashcards, generate_fallback_quiz, generate_fallback_flashcards
from utils.vector_store import vector_store
from utils.export_utils import create_pdf_report
from utils.logger import logger

# ── Background image ────────────────────────────────────────────────────────────
def _load_bg_base64() -> str:
    candidate_paths = [
        ASSETS_DIR / "study_space_bg.jpg",
        Path(__file__).resolve().parent / "assets" / "study_space_bg.jpg",
        Path(r"C:\Users\User\.gemini\antigravity\brain\c5f3c4f8-20e5-471a-944f-668163a58393\study_space_bg_1786033893429.jpg"),
    ]
    for p in candidate_paths:
        try:
            if p.exists():
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except Exception:
            continue
    return ""

_bg = _load_bg_base64()
_bg_css = (
    f'background:linear-gradient(rgba(8,12,24,0.91),rgba(8,12,24,0.96)),url("data:image/jpeg;base64,{_bg}") no-repeat center center fixed;background-size:cover;'
    if _bg else "background:#0a0f1e;"
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Animations ─── */
@keyframes fadeSlideDown  {{ from{{opacity:0;transform:translateY(-24px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes fadeSlideUp    {{ from{{opacity:0;transform:translateY(20px)}}  to{{opacity:1;transform:translateY(0)}} }}
@keyframes floatY         {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}} }}
@keyframes pulseRing      {{ 0%{{box-shadow:0 0 0 0 rgba(99,179,237,.45)}} 70%{{box-shadow:0 0 0 14px rgba(99,179,237,0)}} 100%{{box-shadow:0 0 0 0 rgba(99,179,237,0)}} }}
@keyframes shimmer        {{ 0%{{background-position:-400px 0}} 100%{{background-position:400px 0}} }}
@keyframes gradShift      {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
@keyframes spin           {{ 100%{{transform:rotate(360deg)}} }}

/* ─── Base ─── */
html, body, [class*="css"] {{ font-family:'Inter',sans-serif !important; }}
.stApp {{ {_bg_css} color:#F1F5F9; }}
.block-container {{ padding:1.2rem 2rem 2rem !important; max-width:96% !important; animation:fadeSlideUp .55s cubic-bezier(.16,1,.3,1); }}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,rgba(10,15,30,.9) 0%,rgba(15,20,40,.85) 100%) !important;
    backdrop-filter:blur(20px) !important;
    border-right:1px solid rgba(99,179,237,.18);
}}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2 {{ color:#63B3ED; }}

/* ─── Hero header ─── */
.hero {{
    background:linear-gradient(135deg,rgba(17,25,50,.82) 0%,rgba(20,30,60,.75) 100%);
    backdrop-filter:blur(20px);
    border:1px solid rgba(99,179,237,.3);
    border-radius:20px;
    padding:2rem 2.5rem;
    margin-bottom:1.6rem;
    animation:fadeSlideDown .7s cubic-bezier(.16,1,.3,1);
    position:relative;
    overflow:hidden;
}}
.hero::before {{
    content:'';
    position:absolute;
    inset:0;
    background:linear-gradient(270deg,rgba(99,179,237,.06),rgba(139,92,246,.06),rgba(16,185,129,.06));
    background-size:400% 400%;
    animation:gradShift 8s ease infinite;
    border-radius:20px;
}}
.hero h1 {{
    font-size:2.4rem;
    font-weight:800;
    margin:0;
    background:linear-gradient(135deg,#63B3ED 0%,#9F7AEA 50%,#38B2AC 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    letter-spacing:-.5px;
}}
.hero p {{ color:#94A3B8; margin:.5rem 0 0; font-size:1.05rem; }}

/* ─── Metric cards ─── */
.metric-card {{
    background:rgba(17,25,50,.7);
    backdrop-filter:blur(12px);
    border:1px solid rgba(99,179,237,.2);
    border-radius:14px;
    padding:1.1rem 1.3rem;
    text-align:center;
    transition:all .3s ease;
}}
.metric-card:hover {{ border-color:rgba(99,179,237,.5); transform:translateY(-3px); box-shadow:0 8px 30px rgba(0,0,0,.3); }}
.metric-card .val {{ font-size:1.7rem; font-weight:700; color:#63B3ED; }}
.metric-card .lbl {{ font-size:.8rem; color:#94A3B8; margin-top:.2rem; text-transform:uppercase; letter-spacing:.5px; }}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid rgba(255,255,255,.08); }}
.stTabs [data-baseweb="tab"] {{
    background:rgba(17,25,50,.5);
    border-radius:10px 10px 0 0;
    padding:10px 20px;
    color:#64748B;
    font-weight:600;
    font-size:.9rem;
    transition:all .25s ease;
    border:1px solid transparent;
    border-bottom:none;
}}
.stTabs [data-baseweb="tab"]:hover {{ color:#63B3ED; background:rgba(99,179,237,.08); transform:translateY(-1px); }}
.stTabs [aria-selected="true"] {{
    background:rgba(99,179,237,.18) !important;
    color:#63B3ED !important;
    border-color:rgba(99,179,237,.3) !important;
    border-bottom:3px solid #63B3ED !important;
    font-weight:700 !important;
}}

/* ─── Buttons ─── */
div.stButton > button {{
    border-radius:10px !important;
    font-weight:600 !important;
    transition:all .25s cubic-bezier(.16,1,.3,1) !important;
    font-family:'Inter',sans-serif !important;
}}
div.stButton > button:hover {{ transform:translateY(-2px) scale(1.01) !important; box-shadow:0 6px 22px rgba(99,179,237,.3) !important; }}
div.stButton > button[kind="primary"] {{
    background:linear-gradient(135deg,#2563EB 0%,#1D4ED8 100%) !important;
    border:none !important;
    animation:pulseRing 3.5s infinite;
}}
div.stButton > button[kind="primary"]:hover {{ background:linear-gradient(135deg,#1D4ED8 0%,#1E40AF 100%) !important; }}

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {{ animation:fadeSlideUp .35s ease-out; border-radius:14px !important; transition:transform .2s ease; }}
[data-testid="stChatMessage"]:hover {{ transform:translateX(3px); }}

/* ─── Glassmorphism content blocks ─── */
.glass-box {{
    background:rgba(17,25,50,.65);
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.1);
    border-radius:14px;
    padding:1.4rem 1.6rem;
    margin:1rem 0;
    box-shadow:0 4px 24px rgba(0,0,0,.2);
    transition:all .3s ease;
}}
.glass-box:hover {{ border-color:rgba(99,179,237,.3); box-shadow:0 8px 32px rgba(0,0,0,.3); }}

/* ─── Flashcard ─── */
.flashcard {{
    background:linear-gradient(135deg,rgba(17,25,50,.9) 0%,rgba(10,15,30,.95) 100%);
    backdrop-filter:blur(18px);
    border:2px solid rgba(99,179,237,.5);
    border-radius:22px;
    padding:3rem 2rem;
    min-height:260px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    animation:floatY 5s ease-in-out infinite;
    transition:all .4s ease;
    box-shadow:0 16px 48px rgba(0,0,0,.4),0 0 0 1px rgba(99,179,237,.1);
}}
.flashcard:hover {{ border-color:#9F7AEA; transform:scale(1.02); animation-play-state:paused; }}
.flashcard .badge {{ font-size:.75rem; font-weight:700; color:#64748B; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:1.2rem; }}
.flashcard .content {{ font-size:1.3rem; font-weight:600; color:#63B3ED; line-height:1.6; }}
.flashcard .content.answer {{ color:#34D399; }}

/* ─── Score bar ─── */
.score-bar {{ height:8px; border-radius:99px; background:rgba(255,255,255,.08); margin:.5rem 0 1rem; overflow:hidden; }}
.score-bar-fill {{ height:100%; border-radius:99px; transition:width .8s cubic-bezier(.16,1,.3,1); }}

/* ─── Input / select ─── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background:rgba(17,25,50,.8) !important;
    border:1px solid rgba(99,179,237,.25) !important;
    border-radius:10px !important;
    color:#F1F5F9 !important;
    transition:border-color .2s ease;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{ border-color:rgba(99,179,237,.6) !important; box-shadow:0 0 0 3px rgba(99,179,237,.12) !important; }}

/* ─── Selectbox ─── */
.stSelectbox > div > div {{ background:rgba(17,25,50,.8) !important; border:1px solid rgba(99,179,237,.25) !important; border-radius:10px !important; }}

/* ─── Status badge ─── */
.badge-correct   {{ display:inline-block; padding:.2rem .7rem; background:rgba(52,211,153,.15); color:#34D399; border:1px solid rgba(52,211,153,.3); border-radius:99px; font-size:.8rem; font-weight:600; }}
.badge-incorrect {{ display:inline-block; padding:.2rem .7rem; background:rgba(248,113,113,.15); color:#F87171; border:1px solid rgba(248,113,113,.3); border-radius:99px; font-size:.8rem; font-weight:600; }}
.badge-partial   {{ display:inline-block; padding:.2rem .7rem; background:rgba(251,191,36,.15); color:#FBBF24; border:1px solid rgba(251,191,36,.3); border-radius:99px; font-size:.8rem; font-weight:600; }}

/* ─── Divider ─── */
hr {{ border-color:rgba(255,255,255,.08) !important; }}

/* ─── Success/error/warning custom ─── */
div[data-testid="stAlert"] {{ border-radius:12px !important; backdrop-filter:blur(10px); }}

/* ─── Hide file uploader limit text ─── */
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    display: none !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
_defaults = {
    "chat_history": [],
    "document_text": "",
    "document_metadata": {},
    "flashcards_data": [],
    "current_card_index": 0,
    "card_flipped": False,
    "quiz_questions": [],
    "user_answers": {},
    "summary_output": "",
    "topics_output": "",
    "explain_output": "",
    "rev_plan_output": "",
    "loaded_files": [],       # list of {name, words, pages}
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Ensure LLM API is enabled via backend configuration
os.environ.pop("DISABLE_LLM_API", None)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Study Buddy AI")
    st.caption("Powered by Google Gemini · Your AI Academic Partner")
    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Documents")
    file_source = st.radio("Source", ["Upload Files", "Sample Files"], horizontal=True, label_visibility="collapsed")

    if file_source == "Upload Files":
        st.caption("You can select multiple PDFs and TXT files at once.")
        uploaded_files = st.file_uploader(
            "Upload PDFs or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            load_btn = st.button("📥 Load All Files", type="primary", use_container_width=True, key="btn_load_files")
            if load_btn:
                combined_text = []
                loaded_files = []
                total_pages = 0
                errors = []

                progress = st.progress(0, text="Processing files…")
                for i, f in enumerate(uploaded_files):
                    raw = f.read()
                    ok, size_err = validate_file_size(raw)
                    if not ok:
                        errors.append(f"{f.name}: {size_err}")
                        continue

                    if f.name.lower().endswith(".pdf"):
                        text, meta = extract_text_from_pdf_stream(raw)
                        if "error" in meta:
                            errors.append(f"{f.name}: {meta['error']}")
                            continue
                        pg = meta.get("total_pages", 1)
                    else:
                        text = clean_text(raw.decode("utf-8", errors="ignore"))
                        pg = 1

                    combined_text.append(f"\n\n{'='*60}\nDOCUMENT: {f.name}\n{'='*60}\n\n{text}")
                    wc = len(text.split())
                    total_pages += pg
                    loaded_files.append({"name": f.name, "words": wc, "pages": pg})
                    progress.progress((i + 1) / len(uploaded_files), text=f"Loaded {f.name}")

                progress.empty()

                if combined_text:
                    merged = "\n".join(combined_text)
                    st.session_state.document_text = merged
                    st.session_state.document_metadata = {
                        "word_count": sum(f["words"] for f in loaded_files),
                        "total_pages": total_pages,
                    }
                    st.session_state.loaded_files = loaded_files
                    vector_store.build_index(merged)
                    st.success(f"✅ Loaded **{len(loaded_files)} file(s)** successfully!")

                for err in errors:
                    st.error(err)

        # Show loaded file badges
        if st.session_state.loaded_files:
            st.markdown("**Loaded Files:**")
            for f in st.session_state.loaded_files:
                st.markdown(
                    f"<div style='background:rgba(99,179,237,.12);border:1px solid rgba(99,179,237,.3);"
                    f"border-radius:8px;padding:.35rem .7rem;margin:.25rem 0;font-size:.82rem;'>"
                    f"📄 <b>{f['name']}</b> &nbsp;·&nbsp; {f['words']:,} words &nbsp;·&nbsp; {f['pages']} pg</div>",
                    unsafe_allow_html=True
                )
            if st.button("🗑 Clear All Documents", use_container_width=True, key="btn_clear_docs"):
                st.session_state.document_text = ""
                st.session_state.document_metadata = {}
                st.session_state.loaded_files = []
                st.rerun()
    else:
        sample = st.selectbox("Select sample", ["sample_notes.pdf", "sample_notes.txt"])
        if st.button("Load Sample", use_container_width=True):
            p = SAMPLE_DATA_DIR / sample
            if p.exists():
                raw = p.read_bytes()
                if sample.endswith(".pdf"):
                    text, meta = extract_text_from_pdf_stream(raw)
                else:
                    text = clean_text(raw.decode("utf-8", errors="ignore"))
                    meta = {"word_count": len(text.split()), "total_pages": 1}
                st.session_state.document_text = text
                st.session_state.document_metadata = meta
                st.session_state.loaded_files = [{"name": sample, "words": meta.get("word_count", 0), "pages": meta.get("total_pages", 1)}]
                vector_store.build_index(text)
                st.success(f"✅ Loaded {sample}")
            else:
                st.error("Sample file not found.")

    # ── Document info ─────────────────────────────────────────────────────
    if st.session_state.document_text:
        st.divider()
        meta = st.session_state.document_metadata
        wc   = meta.get("word_count", len(st.session_state.document_text.split()))
        pg   = meta.get("total_pages", 1)
        fc   = len(st.session_state.loaded_files) or 1

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='val'>{fc}</div><div class='lbl'>Files</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='val'>{pg}</div><div class='lbl'>Pages</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='val'>{wc:,}</div><div class='lbl'>Words</div></div>", unsafe_allow_html=True)

        with st.expander("📄 Preview"):
            st.text_area("", value=truncate_text(st.session_state.document_text, 800), height=140, disabled=True, label_visibility="collapsed")

    # ── Controls ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🛠 Controls")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
doc_loaded  = bool(st.session_state.document_text)
_fc         = len(st.session_state.loaded_files)
doc_badge   = (f"🟢 {_fc} file{'s' if _fc != 1 else ''} loaded") if doc_loaded else "⚪ No Document Loaded"

st.markdown(f"""
<div class="hero">
  <h1>🎓 Study Buddy AI</h1>
  <p>Upload your notes — then summarise, quiz, explain, create flashcards, and plan your revision with Google Gemini.</p>
  <p style="margin-top:.8rem;font-size:.85rem;">{doc_badge}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "💬 AI Chat",
    "📑 Summarizer",
    "🎯 Topics & Explainer",
    "📝 Quiz Studio",
    "🎴 Flashcards",
    "📅 Revision Planner",
])
tab_chat, tab_sum, tab_topics, tab_quiz, tab_fc, tab_plan = tabs

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI AGENT CHAT
# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI AGENT CHAT & INSTRUCTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    chat_top_l, chat_top_r = st.columns([4, 1])
    with chat_top_l:
        st.markdown("### 💬 AI Study Assistant & Instruction Hub")
        st.caption("Give direct instructions, ask specific questions about your notes, compare topics, or run tools directly.")
    with chat_top_r:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="btn_clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    quick_prompt = None

    if not doc_loaded:
        st.info("💡 **Tip:** Upload a document in the sidebar to ask questions grounded directly in your study material.")
    else:
        # Quick Instruction Chips
        st.markdown("<div style='font-size:0.82rem;font-weight:600;color:#94A3B8;margin-bottom:0.4rem;'>⚡ QUICK INSTRUCTIONS:</div>", unsafe_allow_html=True)
        q_cols = st.columns(5)
        if q_cols[0].button("📑 Summarize Notes", use_container_width=True, key="qi_sum"):
            quick_prompt = "Summarize the key takeaways from my uploaded notes."
        if q_cols[1].button("🎯 Extract Topics", use_container_width=True, key="qi_top"):
            quick_prompt = "Extract the main topics and key definitions from the document."
        if q_cols[2].button("⚖️ Compare Concepts", use_container_width=True, key="qi_comp"):
            quick_prompt = "Compare the main traditions and concepts mentioned in the notes."
        if q_cols[3].button("❓ Practice Quiz", use_container_width=True, key="qi_quiz"):
            quick_prompt = "Create a 5-question mixed practice quiz from my notes."
        if q_cols[4].button("📅 7-Day Plan", use_container_width=True, key="qi_plan"):
            quick_prompt = "Generate a 7-day revision schedule for these topics."

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle text input or quick prompt click
    user_query = st.chat_input("Type your instruction or question…")
    active_prompt = user_query or quick_prompt

    if active_prompt:
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing instruction with Study Buddy AI…"):
                try:
                    ctx = st.session_state.document_text or (
                        vector_store.retrieve_relevant_context(active_prompt, top_k=5) if vector_store.is_indexed else ""
                    )
                    agent = create_study_buddy_agent(gemini_key=GEMINI_API_KEY)
                    result = agent.invoke({"input": active_prompt, "context": ctx, "chat_history": []})
                    reply = result.get("output", "I couldn't generate a response. Please try again.")
                except Exception as e:
                    logger.error(f"Chat agent error: {e}")
                    reply = f"⚠️ An error occurred: {e}"

            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARIZER
# ═════════════════════════════════════════════════════════════════════════════
with tab_sum:
    st.markdown("### 📑 Document Summarizer")

    if not doc_loaded:
        st.warning("⬅️ Please upload a document in the sidebar first.")
    else:
        meta = st.session_state.document_metadata
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("**Summary Options**")
            depth = st.selectbox("Summary Depth", ["Short", "Medium", "Detailed"], index=1, key="sum_depth")
            st.markdown(f"""
            <div style="margin-top:1rem;font-size:.85rem;color:#64748B;">
            📄 {meta.get('word_count', 0):,} words · {meta.get('total_pages', 1)} pages
            </div>
            """, unsafe_allow_html=True)
            gen_btn = st.button("🚀 Generate Summary", type="primary", use_container_width=True, key="btn_sum")

        with col_right:
            if gen_btn:
                with st.spinner("Generating summary with Gemini…"):
                    from tools.summarizer import NotesSummarizerTool
                    st.session_state.summary_output = NotesSummarizerTool().run({
                        "text": st.session_state.document_text,
                        "summary_type": depth
                    })

            if st.session_state.summary_output:
                st.markdown(st.session_state.summary_output)
                pdf_data = create_pdf_report("Document Summary", st.session_state.summary_output)
                st.download_button("📥 Download PDF", data=pdf_data, file_name="summary.pdf", mime="application/pdf", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOPICS & EXPLAINER
# ═════════════════════════════════════════════════════════════════════════════
with tab_topics:
    col_topics, col_exp = st.columns(2, gap="large")

    # ── Topics ────────────────────────────────────────────────────────────
    with col_topics:
        st.markdown("### 🎯 Key Topic Extraction")
        if not doc_loaded:
            st.warning("Upload a document first.")
        else:
            if st.button("🔍 Extract Topics & Terms", type="primary", use_container_width=True, key="btn_topics"):
                with st.spinner("Extracting topics with Gemini…"):
                    from tools.topic_extractor import TopicExtractionTool
                    st.session_state.topics_output = TopicExtractionTool().run({"text": st.session_state.document_text})

            if st.session_state.topics_output:
                st.markdown(st.session_state.topics_output)
                pdf_data = create_pdf_report("Topics & Key Concepts", st.session_state.topics_output)
                st.download_button("📥 Download PDF", data=pdf_data, file_name="topics.pdf", mime="application/pdf", use_container_width=True)

    # ── Concept Explainer ─────────────────────────────────────────────────
    with col_exp:
        st.markdown("### 💡 Concept Explainer")
        concept_in = st.text_input("Concept to explain", placeholder="e.g. Neural Networks, Photosynthesis…", key="concept_in")
        mode_in = st.selectbox("Explanation Style", [
            "Easy (Beginner-Friendly)",
            "Medium (Standard Overview)",
            "Detailed (Step-by-Step)",
            "Real-Life Examples & Applications",
        ], key="mode_in")

        if st.button("✨ Explain", type="primary", use_container_width=True, key="btn_exp"):
            if not concept_in.strip():
                st.warning("Enter a concept to explain.")
            else:
                with st.spinner(f"Generating '{mode_in}' explanation…"):
                    from tools.concept_explainer import ConceptExplainerTool
                    st.session_state.explain_output = ConceptExplainerTool().run({
                        "concept": concept_in.strip(),
                        "mode": mode_in,
                        "context": st.session_state.document_text,
                    })

        if st.session_state.explain_output:
            st.markdown(st.session_state.explain_output, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUIZ STUDIO
# ═════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.markdown("### 📝 Interactive Quiz Studio")

    q_left, q_right = st.columns([1, 2], gap="large")

    with q_left:
        st.markdown("**Quiz Settings**")
        q_count  = st.slider("Questions", 1, 20, 5, key="q_count")
        q_diff   = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="q_diff")
        q_type   = st.selectbox("Question Type", ["Mixed", "Multiple Choice", "True/False", "Short Answer"], key="q_type")

        if not doc_loaded:
            st.warning("Upload a document first.")

        gen_quiz = st.button("🎲 Generate Quiz", type="primary", use_container_width=True, key="btn_gen_quiz", disabled=not doc_loaded)

    with q_right:
        if gen_quiz and doc_loaded:
            with st.spinner("Building quiz with Gemini…"):
                from tools.quiz_generator import QuizGeneratorTool
                raw_out = QuizGeneratorTool().run({
                    "text": st.session_state.document_text,
                    "num_questions": q_count,
                    "difficulty": q_diff,
                    "question_type": q_type,
                })
                # Try to get structured data for interactive grading
                from utils.helper import safe_json_parse
                import re
                # First try the QuizGeneratorTool directly through fallback
                st.session_state.quiz_questions = generate_fallback_quiz(
                    st.session_state.document_text, num_questions=q_count, difficulty=q_diff, question_type=q_type
                )
                # Re-run with LLM tool output as display string
                st.session_state.quiz_output = raw_out
                st.session_state.user_answers = {}

        if st.session_state.get("quiz_output"):
            st.markdown(st.session_state.quiz_output)

        # ── Interactive grading form ───────────────────────────────────────
        if st.session_state.quiz_questions:
            st.divider()
            st.markdown("### 🎯 Answer & Grade")

            with st.form("quiz_form", clear_on_submit=False):
                submissions = {}
                for i, q in enumerate(st.session_state.quiz_questions, 1):
                    st.markdown(f"**Q{i}. {q.get('question', '')}**")
                    q_t = q.get("type", "Multiple Choice")
                    opts = q.get("options", [])

                    if q_t in ("Multiple Choice", "True/False") and opts:
                        submissions[i] = st.radio(
                            f"Your answer for Q{i}:", opts, index=None, key=f"qa_{i}"
                        )
                    else:
                        submissions[i] = st.text_input(f"Your answer for Q{i}:", key=f"qa_{i}")
                    st.markdown("---")

                submitted = st.form_submit_button("💯 Submit & Grade", type="primary")

            if submitted:
                st.markdown("### 📊 Results")
                correct_count = 0
                total = len(st.session_state.quiz_questions)

                for i, q in enumerate(st.session_state.quiz_questions, 1):
                    user_a = submissions.get(i)
                    correct_a = str(q.get("correct_answer", "")).strip()

                    if not user_a or (isinstance(user_a, str) and not user_a.strip()):
                        st.markdown(f"**Q{i}:** <span class='badge-partial'>Skipped</span> — Correct: *{correct_a}*", unsafe_allow_html=True)
                        continue

                    is_correct = str(user_a).strip().lower() == correct_a.lower() or correct_a.lower() in str(user_a).strip().lower()
                    exp_text = q.get('explanation', '')
                    exp_html = f"<div style='margin-top:0.45rem;padding:0.6rem 0.9rem;background:rgba(99,179,237,0.07);border-left:3px solid #63B3ED;border-radius:6px;font-size:0.88rem;color:#CBD5E1;line-height:1.5;'>💡 {exp_text}</div>" if exp_text else ""
                    
                    if is_correct:
                        correct_count += 1
                        st.markdown(f"**Q{i}:** <span class='badge-correct'>✓ Correct</span>{exp_html}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Q{i}:** <span class='badge-incorrect'>✗ Incorrect</span> — Correct Answer: *{correct_a}*{exp_html}", unsafe_allow_html=True)

                pct = int(correct_count / total * 100) if total else 0
                bar_color = "#34D399" if pct >= 70 else ("#FBBF24" if pct >= 40 else "#F87171")
                emoji = "🏆" if pct >= 90 else ("✅" if pct >= 70 else ("📚" if pct >= 40 else "💡"))

                st.markdown(f"""
                <div style="margin-top:1.5rem;padding:1.5rem;background:rgba(17,25,50,.8);border-radius:14px;border:1px solid rgba(99,179,237,.2);">
                  <div style="font-size:1.5rem;font-weight:800;color:#F1F5F9;">{emoji} Score: {correct_count}/{total} &nbsp; <span style="color:{bar_color};">{pct}%</span></div>
                  <div class="score-bar"><div class="score-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
                  <div style="color:#94A3B8;font-size:.9rem;">{"Excellent work! Keep it up." if pct>=80 else "Good effort. Review the incorrect answers." if pct>=50 else "Keep studying — revisit the material and try again."}</div>
                </div>
                """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — FLASHCARD STUDIO
# ═════════════════════════════════════════════════════════════════════════════
with tab_fc:
    st.markdown("### 🎴 Flashcard Studio")

    fc_left, fc_right = st.columns([1, 2], gap="large")

    with fc_left:
        st.markdown("**Card Settings**")
        fc_count = st.slider("Number of Cards", 3, 20, 6, key="fc_count")
        fc_diff  = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="fc_diff_sel")

        if not doc_loaded:
            st.warning("Upload a document first.")

        gen_fc = st.button("🎴 Generate Flashcards", type="primary", use_container_width=True, key="btn_gen_fc", disabled=not doc_loaded)

        if st.session_state.flashcards_data:
            st.markdown("---")
            st.markdown("**Navigate**")
            b1, b2 = st.columns(2)
            if b1.button("⬅ Prev", use_container_width=True, key="fc_prev", disabled=st.session_state.current_card_index == 0):
                st.session_state.current_card_index -= 1
                st.session_state.card_flipped = False
                st.rerun()
            if b2.button("Next ➡", use_container_width=True, key="fc_next", disabled=st.session_state.current_card_index >= len(st.session_state.flashcards_data) - 1):
                st.session_state.current_card_index += 1
                st.session_state.card_flipped = False
                st.rerun()

            b3, b4 = st.columns(2)
            if b3.button("🔄 Flip", use_container_width=True, key="fc_flip"):
                st.session_state.card_flipped = not st.session_state.card_flipped
                st.rerun()
            if b4.button("🔀 Shuffle", use_container_width=True, key="fc_shuffle"):
                st.session_state.flashcards_data = shuffle_flashcards(st.session_state.flashcards_data)
                st.session_state.current_card_index = 0
                st.session_state.card_flipped = False
                st.rerun()

            # Export
            st.markdown("---")
            st.download_button(
                "📥 Export as JSON",
                data=json.dumps(st.session_state.flashcards_data, indent=2),
                file_name="flashcards.json",
                mime="application/json",
                use_container_width=True,
            )

    with fc_right:
        if gen_fc and doc_loaded:
            with st.spinner("Creating flashcards with Gemini…"):
                from tools.flashcards import FlashcardGeneratorTool
                raw_fc = FlashcardGeneratorTool().run({
                    "text": st.session_state.document_text,
                    "count": fc_count,
                    "difficulty": fc_diff,
                })
                # Also build structured data for interactive viewer
                st.session_state.flashcards_data = generate_fallback_flashcards(
                    st.session_state.document_text, count=fc_count, difficulty=fc_diff
                )
                st.session_state.current_card_index = 0
                st.session_state.card_flipped = False
                st.session_state.fc_output = raw_fc

        if st.session_state.get("fc_output") and not st.session_state.flashcards_data:
            st.markdown(st.session_state.fc_output)

        if st.session_state.flashcards_data:
            cards = st.session_state.flashcards_data
            idx   = st.session_state.current_card_index
            total = len(cards)
            card  = cards[idx]

            flipped  = st.session_state.card_flipped
            content  = card.get("back", "") if flipped else card.get("front", "")
            badge    = "💡 ANSWER" if flipped else "❓ QUESTION"
            cls      = "answer" if flipped else ""

            st.markdown(f"""
            <div class="flashcard">
              <div class="badge">Card {idx+1} / {total} &nbsp;·&nbsp; {card.get('difficulty','Medium')}</div>
              <div class="badge">{badge}</div>
              <div class="content {cls}">{content}</div>
            </div>
            """, unsafe_allow_html=True)

            # Progress dots
            dots = "".join(
                f"<span style='display:inline-block;width:8px;height:8px;border-radius:99px;margin:0 3px;background:{'#63B3ED' if i==idx else 'rgba(255,255,255,.15)'};'></span>"
                for i in range(min(total, 15))
            )
            st.markdown(f"<div style='text-align:center;margin-top:1rem;'>{dots}</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — REVISION PLANNER
# ═════════════════════════════════════════════════════════════════════════════
with tab_plan:
    st.markdown("### 📅 Smart Revision & Study Schedule Planner")
    st.caption("Generate a structured day-by-day revision timetable with active recall checkpoints and Pomodoro sessions.")

    plan_left, plan_right = st.columns([1, 2], gap="large")

    with plan_left:
        st.markdown("**Revision Plan Settings**")
        p_days = st.slider(
            "Plan Duration (Days) — Max 30 Days",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            key="plan_days",
            help="Choose any duration from 1 day up to a maximum limit of 30 days."
        )
        p_hours = st.slider("Daily Study Time (Hours)", min_value=0.5, max_value=8.0, value=2.0, step=0.5, key="plan_hours")
        
        # Auto-extract default topics if document is loaded
        default_topics = ""
        if doc_loaded:
            words = [w for w in st.session_state.document_text.split() if len(w) > 4 and w.isalpha()]
            from collections import Counter
            top_words = [w.title() for w, _ in Counter(words).most_common(6) if w.lower() not in ("which", "their", "about", "there", "these", "would", "other", "after", "document")]
            default_topics = ", ".join(top_words) if top_words else "Core Principles, Problem Solving, Mock Practice"
        else:
            default_topics = "Core Principles, Key Formulas, Problem Solving, Mock Testing"

        p_topics = st.text_area("Topics / Subjects to Cover", value=default_topics, height=100, key="plan_topics", help="Comma-separated topics you want to prioritize.")

        gen_plan = st.button("🚀 Generate Revision Plan", type="primary", use_container_width=True, key="btn_gen_plan")

        # Quick stats summary card
        total_study_hrs = p_days * p_hours
        pomo_cycles = int((total_study_hrs * 60) // 30)
        st.markdown(f"""
        <div class="glass-box" style="margin-top: 1rem; padding: 1rem;">
          <div style="color: #63B3ED; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.5rem;">📊 PLAN PROJECTION</div>
          <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.6;">
            ⏱️ <b>Total Study Hours:</b> {total_study_hrs:.1f} hrs<br>
            🍅 <b>Pomodoro Blocks (25m):</b> {pomo_cycles} cycles<br>
            🎯 <b>Target Mastery:</b> {min(98, 60 + p_days * 2)}%
          </div>
        </div>
        """, unsafe_allow_html=True)

    with plan_right:
        if gen_plan:
            with st.spinner("Crafting customized revision schedule…"):
                from tools.revision_planner import RevisionPlannerTool
                st.session_state.rev_plan_output = RevisionPlannerTool().run({
                    "days": int(p_days),
                    "topics": p_topics,
                    "hours_per_day": float(p_hours)
                })

        if st.session_state.get("rev_plan_output"):
            st.markdown(st.session_state.rev_plan_output)
            pdf_data = create_pdf_report(f"{p_days}-Day Revision Plan", st.session_state.rev_plan_output)
            st.download_button(
                "📥 Download Revision Plan PDF",
                data=pdf_data,
                file_name=f"revision_plan_{p_days}days.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("👈 Choose your revision duration and daily study hours on the left, then click **Generate Revision Plan** to build your custom schedule.")


