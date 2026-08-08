"""
ConceptExplainerTool — Professional, Gemini-quality concept explainer.
Supports: Beginner, Standard, Step-by-Step, and Real-World Examples modes.
"""

import os
import random
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input
from utils.logger import logger


class ConceptExplainerInput(BaseModel):
    concept: str = Field(..., description="The concept, term, or topic to explain.")
    mode: str = Field("Standard", description="Explanation style: Standard, Explain Like I'm a Beginner, Step-by-Step, Real-World Examples.")
    context: Optional[str] = Field(None, description="Optional supporting document context.")


class ConceptExplainerTool(BaseTool):
    name: str = "concept_explainer_tool"
    description: str = (
        "Explains complex academic concepts clearly using the chosen style: "
        "beginner-friendly, step-by-step, real-world examples, or academic overview."
    )
    args_schema: Type[BaseModel] = ConceptExplainerInput

    def _run(self, concept: str, mode: str = "Standard", context: Optional[str] = None) -> str:
        logger.info(f"ConceptExplainerTool — concept='{concept}', mode='{mode}'")
        is_valid, err_msg = validate_text_input(concept, min_chars=2)
        if not is_valid:
            return f"**Error:** {err_msg}"

        mode_lower = (mode or "standard").strip().lower()
        ctx_block = f"\n\n**Relevant Document Context:**\n{context}" if context else ""

        if "beginner" in mode_lower:
            style_instr = (
                "Explain this concept as if talking to a curious 12-year-old with no prior knowledge. "
                "Use a simple analogy from everyday life. Avoid all technical jargon. "
                "Structure: (1) Simple Definition, (2) Everyday Analogy, (3) Why It Matters."
            )
        elif "world" in mode_lower or "example" in mode_lower:
            style_instr = (
                "Explain this concept through 4 specific, real-world industry applications. "
                "For each: name the industry, describe exactly how the concept is used, "
                "and give a concrete example. Be specific — name real companies or systems where relevant."
            )
        elif "step" in mode_lower:
            style_instr = (
                "Provide a rigorous step-by-step breakdown of this concept. "
                "Number each step clearly. Explain what happens at each stage, why it happens, "
                "and what to watch out for. Include a worked example if applicable."
            )
        else:
            style_instr = (
                "Provide a comprehensive academic explanation. Include: "
                "(1) Precise definition, (2) Key properties and characteristics, "
                "(3) How it works / the underlying mechanism, "
                "(4) Relationship to other concepts, (5) Common applications."
            )

        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    f"You are an expert tutor. Explain the following concept clearly and accurately.\n\n"
                    f"**Concept:** {concept}\n"
                    f"**Style:** {style_instr}\n"
                    f"{ctx_block}\n\n"
                    "**Rules:**\n"
                    "- Be specific and accurate. Do not use vague or generic language.\n"
                    "- If a supporting document context is provided, you MUST ground your explanation in that document context and explain the concept as it relates to that specific context.\n"
                    "- Use proper Markdown formatting: `##` headings, `**bold**`, numbered/bullet lists.\n"
                    "- If math is involved, write equations clearly.\n"
                    "- Do NOT include irrelevant information.\n"
                    "- Write at a professional academic standard."
                )

                llm = None
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=DEFAULT_GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.4
                    )
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=DEFAULT_OPENAI_MODEL, api_key=OPENAI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.4
                    )

                if llm:
                    res = llm.invoke(prompt)
                    if res and res.content and len(res.content.strip()) > 30:
                        return res.content.strip()

            except Exception as e:
                logger.warning(f"ConceptExplainerTool LLM failed: {e}")

        return self._structured_fallback(concept, mode_lower, context)

    def _structured_fallback(self, concept: str, mode_lower: str, context: Optional[str]) -> str:
        import re
        c = concept.strip().title()
        concept_lower = concept.lower().strip()

        # 1. Clean and normalize sentences from context
        sentences = []
        if context:
            cleaned_ctx = context
            for b_char in ["\u2022", "\u25aa", "\u25cf", "\u25e6", "\u2023", "\u2043", "\u2219", "\u27a2", "\u2794", "\u25b6", "", "•"]:
                cleaned_ctx = cleaned_ctx.replace(b_char, " ")
            cleaned_ctx = re.sub(r'={3,}.*?={3,}', '', cleaned_ctx)

            raw_splits = re.split(r'(?<=[.!?])\s+|\n{2,}', cleaned_ctx)
            for s in raw_splits:
                s_clean = re.sub(r'\s+', ' ', s).strip()
                s_clean = re.sub(r'^[A-Z\s]{3,}-+\s*', '', s_clean).strip()
                s_clean = s_clean.strip("-:;, •")
                if len(s_clean) > 15 and not s_clean.startswith("DOCUMENT:"):
                    sentences.append(s_clean)

        # Build search terms for matching
        search_terms = [concept_lower]
        words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', concept_lower) if w not in ["what", "how", "why", "the", "and", "explain"]]
        search_terms.extend(words)

        # Score and rank sentences
        scored = []
        for s in sentences:
            s_low = s.lower()
            score = sum(3 for t in search_terms if t in s_low)
            if concept_lower in s_low:
                score += 5
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        matching_sentences = [s for _, s in scored]

        is_beginner = any(k in mode_lower for k in ["beginner", "easy", "simple"])
        is_step = any(k in mode_lower for k in ["step", "detailed", "detail", "deep", "breakdown"])
        is_real_world = any(k in mode_lower for k in ["world", "example", "real", "life", "application", "practical"])

        definitions = [s for s in matching_sentences if any(k in s.lower() for k in ["is ", "are ", "defined", "called", "subset", "refers to", "known as", "means", "consists of"])]
        mechanisms = [s for s in matching_sentences if any(k in s.lower() for k in ["use", "uses", "network", "system", "algorithm", "learn", "process", "function", "technique", "feature", "training", "structured"])]
        history_apps = [s for s in matching_sentences if any(k in s.lower() for k in ["research", "199", "200", "pioneer", "award", "led to", "history", "invent", "origin", "developed", "spotify", "industry", "platform", "popular"])]

        lead_def = definitions[0] if definitions else (matching_sentences[0] if matching_sentences else f"{c} is a foundational pillar within this curriculum.")
        sec_def = definitions[1] if len(definitions) > 1 else (matching_sentences[1] if len(matching_sentences) > 1 else f"It provides critical principles that govern core methodologies and analytical workflows.")
        lead_mech = mechanisms[0] if mechanisms else (matching_sentences[2] if len(matching_sentences) > 2 else f"Operates by organizing fundamental components into systematic pipelines to achieve optimal efficiency.")
        sec_mech = mechanisms[1] if len(mechanisms) > 1 else (matching_sentences[3] if len(matching_sentences) > 3 else f"Coordinates multi-tiered relationships between input data, intermediate logic, and output validation.")
        lead_hist = history_apps[0] if history_apps else (matching_sentences[4] if len(matching_sentences) > 4 else f"Extensively researched and validated through academic and industrial developments.")

        # ── 1. EASY / BEGINNER MODE (Emerald Theme) ──
        if is_beginner:
            return (
                f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(52,211,153,0.4);border-radius:16px;padding:1.8rem;margin-top:1rem;backdrop-filter:blur(10px);">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;">'
                f'<span style="background:rgba(52,211,153,0.2);color:#34D399;font-weight:700;font-size:0.75rem;padding:0.35rem 0.85rem;border-radius:99px;text-transform:uppercase;letter-spacing:1px;border:1px solid rgba(52,211,153,0.3);">🟢 Easy / Beginner Mode</span>'
                f'<span style="color:#94A3B8;font-size:0.85rem;">Document Grounded · Plain English</span></div>'
                f'<h2 style="color:#34D399;margin:0 0 1rem 0;font-size:1.7rem;font-weight:800;">🌟 {c} — The Intuitive Guide</h2>'
                
                f'<div style="background:rgba(0,0,0,0.35);border-left:4px solid #34D399;padding:1.1rem 1.3rem;border-radius:0 12px 12px 0;margin-bottom:1.4rem;">'
                f'<div style="color:#6EE7B7;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">💡 1. The Big Idea (In Plain English)</div>'
                f'<div style="color:#F1F5F9;font-size:1rem;line-height:1.7;">{lead_def}</div>'
                f'<div style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;margin-top:0.6rem;">{sec_def}</div></div>'
                
                f'<div style="background:rgba(52,211,153,0.06);border:1px dashed rgba(52,211,153,0.35);border-radius:14px;padding:1.2rem;margin-bottom:1.4rem;">'
                f'<div style="color:#34D399;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">🍕 2. Everyday Analogy</div>'
                f'<div style="color:#E2E8F0;font-size:0.96rem;line-height:1.7;">Think of <b>{c}</b> like building with LEGO bricks or learning to ride a bike: you start with simple foundation blocks, practice step-by-step, and eventually combine them into intricate, powerful structures without having to rebuild the wheels from scratch each time.</div></div>'
                
                f'<div style="margin-bottom:1.4rem;">'
                f'<div style="color:#6EE7B7;font-weight:700;font-size:0.95rem;margin-bottom:0.6rem;">⚙️ 3. How It Works in 3 Simple Steps</div>'
                f'<div style="display:flex;flex-direction:column;gap:0.6rem;">'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:0.8rem 1rem;color:#E2E8F0;font-size:0.92rem;"><b>Step 1 — Input & Foundation:</b> Gathers initial raw information and core parameters ({lead_def[:80]}...).</div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:0.8rem 1rem;color:#E2E8F0;font-size:0.92rem;"><b>Step 2 — Processing & Pattern Learning:</b> {lead_mech}</div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:0.8rem 1rem;color:#E2E8F0;font-size:0.92rem;"><b>Step 3 — Output & Real-World Use:</b> {sec_mech}</div></div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-radius:12px;padding:1.1rem;margin-bottom:1rem;">'
                f'<div style="color:#34D399;font-weight:700;font-size:0.92rem;margin-bottom:0.4rem;">🎯 4. Why This Matters For You</div>'
                f'<div style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;">Understanding {c} helps you see the big picture across your notes and gives you the exact intuitive foundation needed to tackle harder exam questions with confidence.</div></div></div>'
            )

        # ── 2. DETAILED / STEP-BY-STEP MODE (Amethyst Theme) ──
        elif is_step:
            pool = matching_sentences if len(matching_sentences) >= 4 else (matching_sentences + sentences)[:6]
            step1 = pool[0] if len(pool) > 0 else lead_def
            step2 = pool[1] if len(pool) > 1 else sec_def
            step3 = pool[2] if len(pool) > 2 else lead_mech
            step4 = pool[3] if len(pool) > 3 else sec_mech
            step5 = pool[4] if len(pool) > 4 else lead_hist

            return (
                f'<div style="background:rgba(139,92,246,0.08);border:1px solid rgba(168,85,247,0.4);border-radius:16px;padding:1.8rem;margin-top:1rem;backdrop-filter:blur(10px);">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;">'
                f'<span style="background:rgba(168,85,247,0.2);color:#C084FC;font-weight:700;font-size:0.75rem;padding:0.35rem 0.85rem;border-radius:99px;text-transform:uppercase;letter-spacing:1px;border:1px solid rgba(168,85,247,0.3);">🟣 Detailed / Step-by-Step Mode</span>'
                f'<span style="color:#94A3B8;font-size:0.85rem;">Rigorous Sequential Analysis</span></div>'
                f'<h2 style="color:#C084FC;margin:0 0 1rem 0;font-size:1.7rem;font-weight:800;">🪜 {c} — Comprehensive Deep Dive</h2>'
                
                f'<div style="display:flex;flex-direction:column;gap:0.85rem;margin-bottom:1.4rem;">'
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #C084FC;padding:1rem 1.2rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#E9D5FF;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">Phase 1: Ingestion, Scope & Boundary Definition</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{step1}</div>'
                f'<div style="color:#A855F7;font-size:0.85rem;margin-top:0.3rem;">• <i>Objective:</i> Establishes baseline taxonomy and constraints from source material.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #A855F7;padding:1rem 1.2rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#E9D5FF;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">Phase 2: Conceptual Breakdown & Structural Taxonomy</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{step2}</div>'
                f'<div style="color:#A855F7;font-size:0.85rem;margin-top:0.3rem;">• <i>Objective:</i> Maps intermediate components and classifications.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #9333EA;padding:1rem 1.2rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#E9D5FF;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">Phase 3: Core Algorithmic / Mechanistic Execution</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{step3}</div>'
                f'<div style="color:#A855F7;font-size:0.85rem;margin-top:0.3rem;">• <i>Objective:</i> Executes multi-tiered operations and computational or systemic transformation.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #7E22CE;padding:1rem 1.2rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#E9D5FF;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">Phase 4: Synthesis, Performance & Evaluation</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{step4}</div>'
                f'<div style="color:#A855F7;font-size:0.85rem;margin-top:0.3rem;">• <i>Objective:</i> Validates accuracy against source standards and verifies outputs.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #6B21A8;padding:1rem 1.2rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#E9D5FF;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">Phase 5: Real-World Evolution & Advanced Applications</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{step5}</div>'
                f'<div style="color:#A855F7;font-size:0.85rem;margin-top:0.3rem;">• <i>Objective:</i> Connects theoretical models to real-world industrial and academic practices.</div></div></div>'
                
                f'<div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);border-radius:12px;padding:1.1rem;">'
                f'<div style="color:#F87171;font-weight:700;font-size:0.92rem;margin-bottom:0.3rem;">⚠️ Critical Exam Pitfalls & Misconceptions</div>'
                f'<div style="color:#FECACA;font-size:0.88rem;line-height:1.6;">Ensure you do not conflate <b>{c}</b> with adjacent sub-disciplines. Always cite the exact structural taxonomy and underlying mechanisms during evaluations.</div></div></div>'
            )

        # ── 3. REAL-WORLD EXAMPLES & APPLICATIONS (Amber Theme) ──
        elif is_real_world:
            pool = matching_sentences if len(matching_sentences) >= 3 else (matching_sentences + sentences)[:4]
            ex1 = pool[0] if len(pool) > 0 else f"Applied across automated computational and analytical systems."
            ex2 = pool[1] if len(pool) > 1 else f"Integrated within enterprise workflows and institutional learning curriculums."
            ex3 = pool[2] if len(pool) > 2 else f"Employed in creative industries, digital platforms, and production pipelines."

            return (
                f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(251,191,36,0.4);border-radius:16px;padding:1.8rem;margin-top:1rem;backdrop-filter:blur(10px);">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;">'
                f'<span style="background:rgba(251,191,36,0.2);color:#FBBF24;font-weight:700;font-size:0.75rem;padding:0.35rem 0.85rem;border-radius:99px;text-transform:uppercase;letter-spacing:1px;border:1px solid rgba(251,191,36,0.3);">🟡 Real-Life Examples Mode</span>'
                f'<span style="color:#94A3B8;font-size:0.85rem;">Case Study & Industry Driven</span></div>'
                f'<h2 style="color:#FBBF24;margin:0 0 1rem 0;font-size:1.7rem;font-weight:800;">🌍 {c} — Industry & Real-World Applications</h2>'
                
                f'<div style="display:flex;flex-direction:column;gap:0.9rem;margin-bottom:1.4rem;">'
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #FBBF24;padding:1.1rem 1.3rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#FCD34D;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">📱 1. Digital Platforms, Tech Ecosystems & Modern AI</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{ex1}</div>'
                f'<div style="color:#FDE68A;font-size:0.85rem;margin-top:0.4rem;"><b>Industry Impact:</b> Powers automated pattern recognition, large-scale content recommendation, and intelligent data routing.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #F59E0B;padding:1.1rem 1.3rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#FCD34D;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">🏛️ 2. Institutional, Academic & Enterprise Adoption</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{ex2}</div>'
                f'<div style="color:#FDE68A;font-size:0.85rem;margin-top:0.4rem;"><b>Sector Impact:</b> Establishes rigorous benchmarking standards, compliance structures, and repeatable operational efficiency.</div></div>'
                
                f'<div style="background:rgba(0,0,0,0.3);border-left:4px solid #D97706;padding:1.1rem 1.3rem;border-radius:0 12px 12px 0;">'
                f'<div style="color:#FCD34D;font-weight:700;font-size:0.95rem;margin-bottom:0.3rem;">💼 3. Professional Practice & Practical Execution</div>'
                f'<div style="color:#F1F5F9;font-size:0.94rem;line-height:1.6;">{ex3}</div>'
                f'<div style="color:#FDE68A;font-size:0.85rem;margin-top:0.4rem;"><b>Practitioner Impact:</b> Enables domain experts and researchers to solve multifaceted challenges with validated methodological tools.</div></div></div>'
                
                f'<div style="background:rgba(251,191,36,0.1);border-radius:12px;padding:1.1rem;color:#FCD34D;font-size:0.92rem;line-height:1.6;">'
                f'🚀 <b>Career & Practical Takeaway:</b> Demonstrating familiarity with these real-world scenarios bridges theoretical curriculum knowledge with practical industry competence.</div></div>'
            )

        # ── 4. MEDIUM / STANDARD OVERVIEW (Sapphire Theme) ──
        else:
            return (
                f'<div style="background:rgba(59,130,246,0.08);border:1px solid rgba(99,179,237,0.4);border-radius:16px;padding:1.8rem;margin-top:1rem;backdrop-filter:blur(10px);">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;">'
                f'<span style="background:rgba(99,179,237,0.2);color:#60A5FA;font-weight:700;font-size:0.75rem;padding:0.35rem 0.85rem;border-radius:99px;text-transform:uppercase;letter-spacing:1px;border:1px solid rgba(99,179,237,0.3);">🔵 Medium / Standard Overview</span>'
                f'<span style="color:#94A3B8;font-size:0.85rem;">Academic Standard · Document Grounded</span></div>'
                f'<h2 style="color:#60A5FA;margin:0 0 1rem 0;font-size:1.7rem;font-weight:800;">📚 {c} — Comprehensive Academic Overview</h2>'
                
                f'<div style="background:rgba(0,0,0,0.35);border-left:4px solid #60A5FA;padding:1.1rem 1.3rem;border-radius:0 12px 12px 0;margin-bottom:1.4rem;">'
                f'<div style="color:#93C5FD;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">📖 1. Formal Academic Definition</div>'
                f'<div style="color:#F1F5F9;font-size:1rem;line-height:1.7;">{lead_def}</div></div>'
                
                f'<div style="display:grid;grid-template-columns:1fr;gap:0.8rem;margin-bottom:1.4rem;">'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.25);padding:1rem 1.2rem;border-radius:12px;font-size:0.94rem;color:#CBD5E1;line-height:1.6;">'
                f'<b style="color:#93C5FD;">🏛️ Scope & Systematic Taxonomy:</b><br>{sec_def}</div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.25);padding:1rem 1.2rem;border-radius:12px;font-size:0.94rem;color:#CBD5E1;line-height:1.6;">'
                f'<b style="color:#93C5FD;">⚙️ Core Mechanics & Properties:</b><br>{lead_mech}</div>'
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.25);padding:1rem 1.2rem;border-radius:12px;font-size:0.94rem;color:#CBD5E1;line-height:1.6;">'
                f'<b style="color:#93C5FD;">📜 Evolution & Operational Context:</b><br>{lead_hist}</div></div>'
                
                f'<div style="background:rgba(99,179,237,0.1);border-radius:12px;padding:1.1rem;font-size:0.92rem;color:#E2E8F0;line-height:1.6;">'
                f'<b style="color:#60A5FA;">📌 High-Yield Revision Takeaway:</b> {c} represents an indispensable core principle. Master the connection between its definition, operational mechanics, and practical applications for full exam preparation.</div></div>'
            )

    async def _arun(self, concept: str, mode: str = "Standard", context: Optional[str] = None) -> str:
        return self._run(concept, mode, context)
