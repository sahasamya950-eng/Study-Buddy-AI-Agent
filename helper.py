"""
Helper functions for Study Buddy AI.
Includes high-quality fallback generators, JSON parsing, and shuffling utilities.
"""

import json
import re
import random
from typing import Dict, Any, List, Optional
from utils.logger import logger


# ─────────────────────────────────────────────
# JSON Parsing
# ─────────────────────────────────────────────

def safe_json_parse(text: str) -> Optional[Any]:
    """Robustly extracts and parses JSON from LLM output, handling markdown code blocks."""
    if not text:
        return None
    # Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Strip markdown code fences
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    # Extract first JSON array or object
    for pattern in [r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
# Keyword Extraction
# ─────────────────────────────────────────────

def extract_keywords_from_text(text: str, top_n: int = 10) -> List[str]:
    """Extracts meaningful content words from text (capitalized nouns preferred)."""
    stopwords = {
        "this", "that", "with", "from", "have", "which", "there", "their",
        "about", "would", "these", "other", "could", "into", "some", "been",
        "more", "also", "than", "when", "will", "were", "they", "then",
        "each", "such", "both", "does", "most", "over", "used", "very"
    }
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    filtered = [w.capitalize() for w in words if w.lower() not in stopwords]
    unique = list(dict.fromkeys(filtered))
    return unique[:top_n] if unique else ["Concept", "Principle", "System", "Analysis", "Data"]


# ─────────────────────────────────────────────
# Flashcard Utilities
# ─────────────────────────────────────────────

def shuffle_flashcards(flashcards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cards = list(flashcards)
    random.shuffle(cards)
    return cards


# ─────────────────────────────────────────────
# Fallback Summary
# ─────────────────────────────────────────────

def generate_fallback_summary(text: str, summary_type: str = "Medium") -> str:
    """Extracts actual sentences from the document to produce a grounded fallback summary."""
    if not text or len(text.strip()) < 10:
        return "No document text available to summarise."

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    valid = [s.strip() for s in sentences if 20 < len(s.strip()) < 250]
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]

    top_4 = valid[:4] if valid else ["The document contains academic study material for review."]
    bullet_block = "\n".join(f"- {s}" for s in top_4)

    if summary_type == "Short":
        intro = " ".join(valid[:2]) if len(valid) >= 2 else (valid[0] if valid else "This document covers key academic material.")
        return (
            f"## Summary\n\n{intro}\n\n"
            f"**Key Points:**\n{bullet_block}"
        )
    elif summary_type == "Detailed":
        excerpt = "\n\n> ".join(paras[:4]) if paras else (text[:600] + "...")
        return (
            f"## Comprehensive Summary\n\n"
            f"### Overview\n{' '.join(valid[:3]) if len(valid) >= 3 else (valid[0] if valid else 'Academic document.')}\n\n"
            f"### Key Content\n> {excerpt}\n\n"
            f"### Critical Points\n{bullet_block}\n\n"
            f"### Revision Takeaways\n"
            f"- Identify and define all key terms before attempting practice problems.\n"
            f"- Understand the relationships between main concepts.\n"
            f"- Apply formulas or methodologies to worked examples."
        )
    else:
        return (
            f"## Study Summary\n\n"
            f"### Overview\n{' '.join(valid[:2]) if len(valid) >= 2 else 'Academic study document.'}\n\n"
            f"### Key Points\n{bullet_block}\n\n"
            f"### Revision Notes\n"
            f"- Prioritise understanding core concepts over rote memorisation.\n"
            f"- Connect each topic to practical applications."
        )


# ─────────────────────────────────────────────
# Fallback Topic Extractor
# ─────────────────────────────────────────────

def generate_fallback_topics(text: str) -> Dict[str, Any]:
    """Extracts topics, definitions and formulas from text as a best-effort fallback."""
    import re as _re

    sentences = _re.split(r'(?<=[.!?])\s+', text)
    def_sentences = [s.strip() for s in sentences if any(
        marker in s.lower() for marker in [" is ", " refers to ", " defined as ", " means ", " represents "]
    )]

    capitalized = _re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    unique_caps = list(dict.fromkeys(capitalized))[:8]

    formula_patterns = _re.findall(r'[A-Za-z]\s*=\s*[A-Za-z0-9\s\+\-\*/\(\)]+', text)

    definitions = []
    for s in def_sentences[:4]:
        for marker in [" is ", " refers to ", " defined as ", " means "]:
            if marker in s.lower():
                parts = s.split(marker, 1)
                if len(parts) == 2:
                    definitions.append({"term": parts[0].strip(), "definition": parts[1].strip()})
                    break

    return {
        "main_topics": unique_caps[:4] if unique_caps else ["Core Concepts"],
        "subtopics": unique_caps[4:7] if len(unique_caps) > 4 else ["Fundamentals"],
        "key_terms": unique_caps[:6],
        "definitions": definitions if definitions else [{"term": unique_caps[0] if unique_caps else "Concept", "definition": "A key term from the study material."}],
        "formulas": [f.strip() for f in formula_patterns[:3]] if formula_patterns else [],
        "summary": f"Document covering {', '.join(unique_caps[:3])}." if unique_caps else "Academic study material."
    }


# ─────────────────────────────────────────────
# Fallback Quiz Generator
# ─────────────────────────────────────────────

def generate_fallback_quiz(document_text: str, num_questions: int = 5, difficulty: str = "Medium", question_type: str = "Mixed") -> List[Dict[str, Any]]:
    """Generates precise, document-grounded quiz questions across Multiple Choice, True/False, Short Answer, and Mixed modes."""
    raw_lines = [line.strip() for line in document_text.split('\n') if line.strip() and not line.startswith("===")]
    sentences = []
    for line in raw_lines:
        parts = re.split(r'(?<=[.!?])\s+', line)
        for p in parts:
            p_clean = p.strip()
            if 18 < len(p_clean) < 300 and not p_clean.startswith("---") and not p_clean.startswith("Page"):
                sentences.append(p_clean)

    keywords = extract_keywords_from_text(document_text, top_n=15)
    
    mcq_pool = []
    tf_pool = []
    sa_pool = []

    def _make_mcq(q_text, correct, distractors, explanation):
        opts = [correct]
        for d in distractors:
            if d != correct and d not in opts and len(d.strip()) > 0:
                opts.append(d)
        while len(opts) < 4:
            opts.append(f"Alternative Option {len(opts) + 1}")
        opts = opts[:4]
        random.shuffle(opts)
        return {
            "question": q_text, "options": opts,
            "correct_answer": correct, "explanation": explanation,
            "type": "Multiple Choice", "difficulty": difficulty
        }

    def _make_tf(q_text, answer, explanation):
        return {
            "question": q_text, "options": ["True", "False"],
            "correct_answer": answer, "explanation": explanation,
            "type": "True/False", "difficulty": difficulty
        }

    def _make_sa(q_text, answer, explanation):
        return {
            "question": q_text, "options": [],
            "correct_answer": answer, "explanation": explanation,
            "type": "Short Answer", "difficulty": difficulty
        }

    # ── 1. BUILD MCQ POOL ──
    # A. Classifications & Divisions
    for s in sentences:
        for marker in ["is divided into", "are divided into", "consists of", "comprises of", "classified into"]:
            if marker in s.lower():
                parts = re.split(re.escape(marker), s, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    subj = re.sub(r'\s+(and|that|which|is|are|and is)$', '', parts[0].strip(), flags=re.IGNORECASE).rstrip(":,")
                    details = parts[1].strip().rstrip(".:")
                    mcq_pool.append(_make_mcq(
                        f"According to the notes, what {marker} {subj}?",
                        details,
                        [
                            "A single unified category without sub-divisions",
                            "Strictly modern synthetic and electronic variations only",
                            "Uncategorized regional practices without formal rules"
                        ],
                        f"**Conceptual Explanation & Evidence:**\n• ✅ **Correct Answer ({details}):** The study notes explicitly record that {subj} {marker} {details}.\n• ❌ **Distractor Breakdown:** The incorrect options introduce inaccurate generalizations or modern assumptions not supported by your uploaded notes."
                    ))
                    break

    # B. Naming & Definitions
    for s in sentences:
        for marker in ["is called", "are called", "is known as", "known as", "refers to"]:
            if marker in s.lower():
                parts = re.split(re.escape(marker), s, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    subj = re.sub(r'\s+(and|that|which|is|are|and is)$', '', parts[0].strip(), flags=re.IGNORECASE).rstrip(":,")
                    val = parts[1].strip().rstrip(".:")
                    other_kws = [k for k in keywords if k.lower() not in val.lower()][:3]
                    distractors = other_kws if len(other_kws) >= 3 else ["Contemporary Form", "Standard Practice", "Alternative Style"]
                    mcq_pool.append(_make_mcq(
                        f"In the uploaded text, what {marker} {subj}?",
                        val,
                        distractors,
                        f"**Definition & Terminology Context:**\n• ✅ **Correct Term ({val}):** In academic terminology, {subj} is specifically designated as {val}.\n• 💡 **Study Significance:** Recognizing this precise term clarifies its foundational identity and prevents confusion with other regional traditions."
                    ))
                    break

    # C. Characteristics
    for s in sentences:
        for marker in ["is characterized by", "characterized by", "is characterised by"]:
            if marker in s.lower():
                parts = re.split(re.escape(marker), s, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    subj = re.sub(r'\s+(and|that|which|is|are|and is)$', '', parts[0].strip(), flags=re.IGNORECASE).rstrip(":,")
                    charac = parts[1].strip().rstrip(".:")
                    mcq_pool.append(_make_mcq(
                        f"What characterizes {subj} according to the document?",
                        charac,
                        [
                            "Strict adherence to modern Western electronic notation",
                            "Complete absence of historical cultural traditions",
                            "Uniform global patterns without individual melodies"
                        ],
                        f"**Characteristic & Analysis:**\n• ✅ **Core Trait:** The material highlights {subj} by emphasizing {charac}.\n• 💡 **Significance:** These distinctive qualities provide the core structural and stylistic framework outlined in the study curriculum."
                    ))
                    break

    # D. Statement Comprehension MCQs
    for s in sentences[:8]:
        if 40 < len(s) < 180 and not any(p["explanation"].startswith(f"**Fact Verification:** {s[:30]}") for p in mcq_pool):
            mcq_pool.append(_make_mcq(
                f"Which of the following statements is directly confirmed by the uploaded study material?",
                s,
                [
                    "The text explicitly rejects all traditional classifications.",
                    "The document contains no references to historical or regional variations.",
                    "The notes state that modern practices completely replaced earlier forms."
                ],
                f"**Fact Verification & Context:**\n• ✅ **Verified Fact:** \"{s}\"\n• ❌ **Analysis:** The alternative choices present common misconceptions or contradictory statements that are refuted by the uploaded text."
            ))

    # ── 2. BUILD TRUE / FALSE POOL ──
    for s in sentences[:25]:
        if len(s) > 28:
            tf_pool.append(_make_tf(
                f"True or False: According to the notes, \"{s}\"",
                "True",
                f"**Evidence & Explanation:**\n• ✅ **True:** The uploaded study notes explicitly validate this statement: \"{s}\". This reinforces essential factual knowledge required for exam mastery."
            ))
    # Add plausible False questions based on document context
    if keywords:
        kw = keywords[0]
        tf_pool.append(_make_tf(
            f"True or False: The uploaded notes state that {kw} has no historical or cultural relevance in the subject.",
            "False",
            f"**Evidence & Correction:**\n• ❌ **False:** The study notes recognize {kw} as a foundational and active topic. The claim that it has no relevance contradicts the uploaded material."
        ))
    if len(keywords) > 2:
        kw2 = keywords[1]
        tf_pool.append(_make_tf(
            f"True or False: The document claims that {kw2} was completely abolished in the modern era.",
            "False",
            f"**Evidence & Correction:**\n• ❌ **False:** The uploaded material demonstrates that {kw2} continues to be studied and practiced as an ongoing pillar of the curriculum."
        ))

    # ── 3. BUILD SHORT ANSWER POOL ──
    for s in sentences:
        for marker in ["is divided into", "is called", "characterized by", "consists of", "refers to"]:
            if marker in s.lower():
                parts = re.split(re.escape(marker), s, flags=re.IGNORECASE)
                if len(parts) == 2 and len(parts[0].strip()) > 3:
                    subj = re.sub(r'\s+(and|that|which|is|are|and is)$', '', parts[0].strip(), flags=re.IGNORECASE).rstrip(":,")
                    sa_pool.append(_make_sa(
                        f"Explain what the notes state regarding '{subj}' and its core {marker.replace('is ', '')}.",
                        s,
                        f"**Model Solution & Scoring Rubric:**\n• **Core Answer:** \"{s}\"\n• **Evaluation Focus:** Full marks require identifying '{subj}' and explaining its specific {marker.replace('is ', '')} as detailed in the document."
                    ))
                    break

    for kw in keywords[:5]:
        if not any(kw in q["question"] for q in sa_pool):
            sa_pool.append(_make_sa(
                f"In 1-2 sentences, describe the role and key context of '{kw}' based on the uploaded notes.",
                f"{kw} is highlighted in the text as a significant topic with specific definitions and applications.",
                f"**Grading Key & Context:**\n• State the definition and context of '{kw}'.\n• Connect '{kw}' to the broader themes and regional expressions present in your study material."
            ))

    # Fallback if pools are sparse
    if not mcq_pool:
        for kw in keywords[:5]:
            mcq_pool.append(_make_mcq(
                f"Which key concept is explored in the uploaded study notes?",
                kw,
                ["General Unrelated Topic", "External Literature", "Hypothetical Scenario"],
                f"'{kw}' is highlighted as a core topic in the document."
            ))

    # ── FILTER & ASSEMBLE BASED ON REQUESTED TYPE ──
    q_type_lower = question_type.lower()
    selected_pool = []

    if "choice" in q_type_lower or "mcq" in q_type_lower:
        random.shuffle(mcq_pool)
        selected_pool = mcq_pool[:num_questions]
    elif "true" in q_type_lower or "false" in q_type_lower:
        random.shuffle(tf_pool)
        selected_pool = tf_pool[:num_questions]
    elif "short" in q_type_lower or "answer" in q_type_lower:
        random.shuffle(sa_pool)
        selected_pool = sa_pool[:num_questions]
    else:
        # MIXED MODE: combine MCQs (50%), True/False (30%), Short Answer (20%)
        random.shuffle(mcq_pool)
        random.shuffle(tf_pool)
        random.shuffle(sa_pool)

        n_mcq = max(1, int(num_questions * 0.5))
        n_tf = max(1, int(num_questions * 0.3))
        n_sa = max(1, num_questions - (n_mcq + n_tf))

        mixed_list = mcq_pool[:n_mcq] + tf_pool[:n_tf] + sa_pool[:n_sa]
        random.shuffle(mixed_list)
        selected_pool = mixed_list[:num_questions]

    # Assign IDs
    for idx, q in enumerate(selected_pool, start=1):
        q["id"] = idx

    return selected_pool


# ─────────────────────────────────────────────
# Fallback Flashcard Generator
# ─────────────────────────────────────────────

def generate_fallback_flashcards(document_text: str, count: int = 5, difficulty: str = "Medium") -> List[Dict[str, str]]:
    """Generates document-grounded fallback flashcards using sentence and keyword extraction."""
    sentences = re.split(r'(?<=[.!?])\s+', document_text.strip())
    def_sentences = [s.strip() for s in sentences if any(
        m in s.lower() for m in [" is ", " refers to ", " defined as ", " means "]
    ) and 20 < len(s.strip()) < 200]
    keywords = extract_keywords_from_text(document_text, top_n=10)

    cards = []

    # Create definition-based cards from document sentences
    for s in def_sentences[:4]:
        for marker in [" is ", " refers to ", " defined as ", " means "]:
            if marker in s.lower():
                parts = s.split(marker, 1)
                if len(parts) == 2 and len(parts[0].strip()) > 2:
                    cards.append({
                        "front": f"What is {parts[0].strip()}?",
                        "back": parts[1].strip().rstrip(".") + ".",
                        "difficulty": difficulty
                    })
                    break

    # Fill with keyword-based cards
    for kw in keywords:
        if len(cards) >= count:
            break
        cards.append({
            "front": f"What is the significance of **{kw}** in this subject?",
            "back": f"{kw} is a key concept from the study material that plays an important role in the subject's core framework and analysis.",
            "difficulty": difficulty
        })

    # General knowledge filler if still short
    general = [
        {"front": "What is active recall?", "back": "A study technique where you actively retrieve information from memory rather than passively re-reading it.", "difficulty": difficulty},
        {"front": "What is spaced repetition?", "back": "A learning method where material is reviewed at increasing intervals over time to improve long-term retention.", "difficulty": difficulty},
        {"front": "What is the Pomodoro technique?", "back": "A time management method using 25-minute focused study intervals separated by 5-minute breaks.", "difficulty": difficulty},
    ]
    for g in general:
        if len(cards) >= count:
            break
        cards.append(g)

    random.shuffle(cards)
    return cards[:count]
