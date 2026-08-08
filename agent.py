"""
LangChain AI Agent orchestration for Study Buddy AI.
Supports tool calling, conversational memory, and resilient fallback execution.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    try:
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        from langchain.agents import create_tool_calling_agent
        from langchain.agents.agent import AgentExecutor

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.pdf_reader import PDFReaderTool
from tools.summarizer import NotesSummarizerTool
from tools.topic_extractor import TopicExtractionTool
from tools.quiz_generator import QuizGeneratorTool
from tools.evaluator import AnswerEvaluationTool
from tools.flashcards import FlashcardGeneratorTool
from tools.revision_planner import RevisionPlannerTool
from tools.concept_explainer import ConceptExplainerTool
from tools.difficulty import DifficultyLevelTool

from utils.constants import SYSTEM_PROMPT_TEMPLATE
from utils.logger import logger
from config import GEMINI_API_KEY, OPENAI_API_KEY

def get_all_study_tools() -> List[BaseTool]:
    """Returns an instantiated list of 9 custom tools."""
    return [
        PDFReaderTool(),
        NotesSummarizerTool(),
        TopicExtractionTool(),
        QuizGeneratorTool(),
        AnswerEvaluationTool(),
        FlashcardGeneratorTool(),
        RevisionPlannerTool(),
        ConceptExplainerTool(),
        DifficultyLevelTool(),
    ]

class FallbackStudyAgent:
    """Fallback agent for deterministic intent routing when LLM API keys are unavailable."""

    def __init__(self, tools: List[BaseTool]):
        self.tools_map = {tool.name: tool for tool in tools}

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import re
        query = str(inputs.get("input", "")).strip()
        context = str(inputs.get("context", "")).strip()
        query_lower = query.lower()

        logger.info(f"FallbackStudyAgent processing instruction: {query[:60]}")

        # ── 1. QUIZ & QUESTION GENERATION ──
        is_quiz_request = any(w in query_lower for w in [
            "quiz", "mcq", "mcqs", "true false", "true/false", "test me", "practice question", "practice questions",
            "make question", "make questions", "create question", "create questions", "generate question", "generate questions",
            "give question", "give questions", "give me question", "give me questions", "questions with answer", "questions and answer"
        ]) or (("question" in query_lower or "questions" in query_lower) and any(verb in query_lower for verb in ["make", "give", "create", "generate", "build", "provide", "show", "prepare", "set"]))

        if is_quiz_request:
            tool = self.tools_map["quiz_generator_tool"]
            if "true" in query_lower or "false" in query_lower or "t/f" in query_lower:
                q_type = "True/False"
            elif "mcq" in query_lower or "multiple choice" in query_lower or "choice" in query_lower:
                q_type = "Multiple Choice"
            elif "short" in query_lower:
                q_type = "Short Answer"
            else:
                q_type = "Mixed"

            num_match = re.search(r'\b(\d+)\s*(?:questions?|mcqs?|items?)\b', query_lower)
            num_q = min(20, max(1, int(num_match.group(1)))) if num_match else 5
            return {"output": tool.run({"text": context or query, "num_questions": num_q, "difficulty": "Medium", "question_type": q_type})}

        # ── 2. FLASHCARDS & REVISION PLANS ──
        if any(w in query_lower for w in ["flashcard", "revision card", "make card"]):
            tool = self.tools_map["flashcard_generator_tool"]
            return {"output": tool.run({"text": context or query, "count": 6, "difficulty": "Medium"})}

        if any(w in query_lower for w in ["revision plan", "study schedule", "timetable", "study plan"]):
            tool = self.tools_map["revision_planner_tool"]
            return {"output": tool.run({"days": 7, "topics": query, "hours_per_day": 2.0})}

        # ── 3. SUMMARIZATION & OVERVIEWS ──
        if any(w in query_lower for w in ["summarize", "summarise", "summary", "key takeaways", "overview", "recap", "synopsis"]):
            tool = self.tools_map["notes_summarizer_tool"]
            if "detailed" in query_lower or "comprehensive" in query_lower or "deep" in query_lower:
                s_type = "Detailed"
            elif "short" in query_lower or "brief" in query_lower or "concise" in query_lower:
                s_type = "Short"
            else:
                s_type = "Medium"
            return {"output": tool.run({"text": context or query, "summary_type": s_type})}

        # ── 4. TOPIC & FORMULA EXTRACTION ──
        if any(w in query_lower for w in ["extract topic", "topics", "formulas", "definitions", "glossary", "key terms"]):
            tool = self.tools_map["topic_extraction_tool"]
            return {"output": tool.run({"text": context or query})}

        # ── 5. DOCUMENT-GROUNDED INSTRUCTION PROCESSING ──
        if context:
            stopwords = {
                "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
                "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "having", "the", "a", "an", "and", "or", "but", "if",
                "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                "against", "between", "into", "through", "during", "before", "after", "above",
                "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                "again", "further", "then", "once", "here", "there", "all", "any", "both",
                "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
                "should", "now", "tell", "give", "me", "please", "find", "list", "show",
                "write", "create", "make", "generate", "based", "on", "my", "notes", "uploaded",
                "document", "text", "information"
            }
            query_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', query) if w.lower() not in stopwords]

            # Clean text of bullet characters, stray hyphens, and joined fragments
            cleaned_context = context
            for b_char in ["\u2022", "\u25aa", "\u25cf", "\u25e6", "\u2023", "\u2043", "\u2219", "\u27a2", "\u2794", "\u25b6", "", "•"]:
                cleaned_context = cleaned_context.replace(b_char, " ")
            
            # Clean header underlines like === DOCUMENT: filename ===
            cleaned_context = re.sub(r'={3,}.*?={3,}', '', cleaned_context)

            # Split sentences intelligently
            raw_splits = re.split(r'(?<=[.!?])\s+|\n{2,}', cleaned_context)
            raw_sentences = []
            for s in raw_splits:
                # Merge multi-line broken sentences
                s_clean = re.sub(r'\s+', ' ', s).strip()
                # Clean page numbers and document headers at start of sentence
                s_clean = re.sub(r'^(?:Page\s*\d+\s*[-—–]*\s*)+', '', s_clean, flags=re.IGNORECASE).strip()
                s_clean = re.sub(r'^(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*\s+)+', '', s_clean).strip()
                s_clean = re.sub(r'^[A-Z0-9\s/–—-]{3,}\s*[-—–]+\s*', '', s_clean).strip()
                s_clean = re.sub(r'^(?:Chapter\s*\d+|Section\s*\d+|Page\s*\d+)\s*[:—–-]*\s*', '', s_clean, flags=re.IGNORECASE).strip()
                s_clean = s_clean.strip("-:;, •")
                if len(s_clean) > 20 and not s_clean.startswith("DOCUMENT:"):
                    # Deduplicate near-identical sentences
                    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s_clean.lower()))
                    if words:
                        is_dup = False
                        for existing in raw_sentences:
                            e_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', existing.lower()))
                            if len(words & e_words) / max(1, min(len(words), len(e_words))) > 0.70:
                                is_dup = True
                                break
                        if not is_dup:
                            raw_sentences.append(s_clean)

            # Rank sentences by keyword relevance
            scored_sentences = []
            for s in raw_sentences:
                s_lower = s.lower()
                score = sum(2 for w in query_words if w in s_lower)
                if len(query_words) >= 2 and " ".join(query_words[:2]) in s_lower:
                    score += 5
                if score > 0:
                    scored_sentences.append((score, s))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            matched = [s for _, s in scored_sentences[:10]]
            corpus = matched if matched else raw_sentences[:8]

            # ── A0. MULTI-QUESTION PARSER & INDIVIDUAL DETAILED ANSWERS ──
            q_segments = [s.strip() for s in re.split(r'(?:^\d+[\.\)]\s*|\s+\d+[\.\)]\s*|\n\d+[\.\)]\s*|\bQ\d+[\.\:]\s*)', query) if len(s.strip()) > 3]
            if len(q_segments) < 2 and query.count("?") >= 2:
                q_segments = [q.strip() + "?" for q in query.split("?") if len(q.strip()) > 5]

            if q_segments and len(q_segments) >= 2:
                lines = [f"## 💡 Detailed & Precise Answers to All Questions\n"]
                for idx, sub_q in enumerate(q_segments, 1):
                    sub_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', sub_q) if w.lower() not in stopwords]
                    sub_scored = []
                    for s in raw_sentences:
                        s_lower = s.lower()
                        score = sum(2 for w in sub_words if w in s_lower)
                        if len(sub_words) >= 2 and " ".join(sub_words[:2]) in s_lower:
                            score += 5
                        if score > 0:
                            sub_scored.append((score, s))
                    sub_scored.sort(key=lambda x: x[0], reverse=True)
                    sub_match = [s for _, s in sub_scored[:3]]
                    
                    cleaned_sub = []
                    for s in sub_match:
                        c = re.sub(r'^[A-Z\s]{3,}-+\s*', '', s).strip()
                        c = c.strip("-:;, ")
                        if len(c) > 10 and c not in cleaned_sub:
                            cleaned_sub.append(c)

                    lines.append(f"### ❓ Question {idx}: *{sub_q.strip()}*\n")
                    if cleaned_sub:
                        lines.append(f"{cleaned_sub[0]}")
                        if len(cleaned_sub) > 1:
                            lines.append(f"\n{cleaned_sub[1]}")
                    else:
                        fallback_c = raw_sentences[min(idx-1, len(raw_sentences)-1)].strip("-:;, ")
                        lines.append(f"{fallback_c}")
                    lines.append("")
                return {"output": "\n".join(lines)}

            # ── A1. COMPREHENSIVE WHOLE DOCUMENT ANALYSIS ──
            if any(w in query_lower for w in ["analyze", "analysis", "deep dive", "breakdown", "whole notes", "full notes", "examine", "comprehensive", "precisely", "overview of whole"]):
                word_count = len(context.split())
                sent_count = len(raw_sentences)
                est_minutes = max(1, word_count // 150)
                
                # Extract key definitions
                def_sentences = [s for s in raw_sentences if any(m in s.lower() for m in ["is ", "are ", "called", "known as", "refers to", "means", "divided into"])]
                numeric_sentences = [s for s in raw_sentences if re.search(r'\b\d+\b', s)]
                
                lines = [
                    f"## 📑 Comprehensive Document Analysis & Precision Report\n",
                    f"| Metric | Document Value | Study Assessment |",
                    f"| :--- | :--- | :--- |",
                    f"| **Total Word Count** | ~{word_count} words | Standard Academic Unit |",
                    f"| **Key Sentence Count** | {sent_count} foundational sentences | High Information Density |",
                    f"| **Estimated Reading Time** | ~{est_minutes} min | Recommended 25-min Pomodoro Block |\n",
                    f"### 🎯 1. Executive Summary & Core Subject Domain",
                    f"{raw_sentences[0] if raw_sentences else 'The study material provides structured academic coverage of the subject matter.'} "
                    f"{raw_sentences[1] if len(raw_sentences) > 1 else ''}\n",
                    f"### 🏛️ 2. Core Thematic Pillars & Taxonomy\n",
                ]
                
                for i, s in enumerate(raw_sentences[2:6] if len(raw_sentences) > 5 else raw_sentences[:4], 1):
                    lines.append(f"* **Pillar {i}:** {s}\n")
                
                lines.append("### 📊 3. Key Terminology & Conceptual Framework\n")
                lines.append("| Academic Term / Topic | Exact Definition / Context from Notes | Syllabus Role |")
                lines.append("| :--- | :--- | :--- |")
                
                table_pool = def_sentences if def_sentences else raw_sentences
                for i, s in enumerate(table_pool[:5], 1):
                    parts = re.split(r'\s+(is|are|called|divided into|characterized by|has)\s+', s, maxsplit=1, flags=re.IGNORECASE)
                    term = parts[0][:35] if parts else f"Concept {i}"
                    desc = parts[2][:75] if len(parts) > 2 else s[:75]
                    lines.append(f"| **{term}** | {desc} | Core examinable concept |")
                
                if numeric_sentences:
                    lines.append("\n### 🔢 4. Quantitative Data, Statistics & Named Citations\n")
                    for i, s in enumerate(numeric_sentences[:4], 1):
                        lines.append(f"{i}. **Data Point:** {s}\n")
                
                lines.append("### 📌 5. High-Yield Revision Takeaways for Mastery\n")
                for i, s in enumerate(raw_sentences[:5], 1):
                    lines.append(f"* **Key Takeaway {i}:** {s}\n")
                
                lines.append("\n---\n💡 **Recommended Next Steps:**\n")
                lines.append("* Type `make true false quiz questions` to test your active recall.")
                lines.append("* Type `make a 7-day study plan` to schedule your revision intervals.")
                lines.append("\n🔍 *Analysis generated with precision directly from your uploaded document.*")
                
                return {"output": "\n".join(lines)}

            # A. TABLE INSTRUCTION ("table", "tabular", "columns", "grid")
            if any(w in query_lower for w in ["table", "tabular", "column", "grid"]):
                lines = [
                    f"### 📊 Structured Table: {query.title()}\n",
                    "| Item / Concept | Key Details & Classification from Notes | Significance |",
                    "| :--- | :--- | :--- |"
                ]
                for i, s in enumerate(corpus[:5], 1):
                    parts = re.split(r'\s+(is|are|called|divided into|characterized by|has)\s+', s, maxsplit=1, flags=re.IGNORECASE)
                    term = parts[0][:30] if parts else f"Concept {i}"
                    desc = parts[2][:70] if len(parts) > 2 else s[:70]
                    lines.append(f"| **{term}** | {desc} | Core curriculum takeaway |")
                lines.append("\n*Table generated directly from your uploaded document content.*")
                return {"output": "\n".join(lines)}

            # B. ESSAY / DETAILED WRITEUP ("essay", "article", "paragraph", "write about", "detailed writeup", "in detail")
            if any(w in query_lower for w in ["essay", "article", "paragraph", "write about", "detailed writeup", "in detail"]):
                lines = [
                    f"### 📝 Detailed Analysis & Essay: {query.title()}\n",
                    f"#### 1. Introduction & Foundational Overview\n{corpus[0] if corpus else 'The study material outlines foundational principles essential for understanding this topic.'}\n",
                    f"#### 2. Core Concepts & Thematic Breakdown\n" + "\n\n".join(corpus[1:4] if len(corpus) > 3 else corpus),
                    f"\n#### 3. Synthesis & Academic Takeaway\n" + (corpus[4] if len(corpus) > 4 else "Understanding these relationships provides critical analytical context for mastery and examination success.") + "\n",
                    "---\n*Synthesized directly from your uploaded study notes.*"
                ]
                return {"output": "\n".join(lines)}

            # C. COMPARISON INSTRUCTION ("compare", "difference", " vs ", "versus", "distinguish", "contrast")
            if any(w in query_lower for w in ["compare", "difference", " vs ", "versus", "distinguish", "contrast"]):
                lines = [
                    f"### ⚖️ Comparative Breakdown: {query.title()}\n",
                    "| Feature / Aspect | First Tradition / Concept | Second Tradition / Concept |",
                    "| :--- | :--- | :--- |"
                ]
                if len(corpus) >= 2:
                    lines.append(f"| **Origin & Classification** | {corpus[0][:60]} | {corpus[1][:60]} |")
                if len(corpus) >= 4:
                    lines.append(f"| **Stylistic Characteristics** | {corpus[2][:60]} | {corpus[3][:60]} |")
                lines.append("\n**Key Distinctions from Your Notes:**\n")
                for i, s in enumerate(corpus[:4], 1):
                    lines.append(f"* **Point {i}:** {s}\n")
                lines.append("💡 *Both elements represent complementary pillars within the subject curriculum.*")
                return {"output": "\n".join(lines)}

            # D. DATA & NUMBERS EXTRACTION ("number", "date", "statistic", "data", "count", "figure")
            if any(w in query_lower for w in ["number", "numbers", "date", "dates", "statistic", "statistics", "data", "figure", "figures", "how many"]):
                numeric_sentences = [s for s in raw_sentences if re.search(r'\b\d+\b', s)]
                lines = [
                    f"### 🔢 Facts, Figures & Data Extracted from Notes\n",
                    "Here are the specific quantitative facts and statistics recorded in your study material:\n"
                ]
                target_nums = numeric_sentences if numeric_sentences else corpus[:4]
                for i, s in enumerate(target_nums[:6], 1):
                    lines.append(f"{i}. **Data Point:** {s}\n")
                return {"output": "\n".join(lines)}

            # E. GLOSSARY / DEFINITIONS ("glossary", "define", "definitions", "terms", "vocabulary")
            if any(w in query_lower for w in ["glossary", "define", "definition", "definitions", "term", "terms", "vocabulary"]):
                def_sentences = [s for s in raw_sentences if any(m in s.lower() for m in ["is ", "called", "known as", "refers to", "means"])]
                lines = [
                    f"### 📖 Document Glossary & Definitions\n",
                    "Key academic terminology defined in your uploaded notes:\n"
                ]
                target_defs = def_sentences if def_sentences else corpus[:5]
                for i, s in enumerate(target_defs[:6], 1):
                    lines.append(f"* **Definition {i}:** {s}\n")
                return {"output": "\n".join(lines)}

            # F. LIST / BULLET POINTS / POINT WISE ("list", "bullet", "points", "point", "steps", "step", "outline", "summarize", "summary", "key takeaways", "point wise", "pointwise")
            if any(w in query_lower for w in ["list", "bullet", "points", "point", "steps", "step", "outline", "summarize", "summary", "key takeaways", "point wise", "pointwise"]):
                lines = [
                    f"### 📋 Key Takeaways & Point-by-Point Analysis\n",
                    "Here are the core takeaways extracted directly from your study notes, structured in individual point-wise paragraphs:\n"
                ]
                for i, s in enumerate(corpus[:6], 1):
                    words = s.split()
                    heading = " ".join(words[:4]).rstrip(",.:;") if len(words) >= 4 else f"Core Concept {i}"
                    lines.append(f"* **Point {i} ({heading}):**  \n  {s}\n")
                lines.append("💡 *Ask follow-up questions to dive deeper into any specific point.*")
                return {"output": "\n".join(lines)}

            # ── H. BROAD, COMPREHENSIVE CONCEPTUAL EXPLANATION & ACCURATE Q&A ──
            if matched:
                clean_matches = []
                for s in matched:
                    cleaned = re.sub(r'^[A-Z\s]{3,}-+\s*', '', s).strip()
                    cleaned = cleaned.strip("-:;, •")
                    if len(cleaned) > 15 and cleaned not in clean_matches:
                        clean_matches.append(cleaned)

                primary_subject = " ".join([w.title() for w in query_words[:3]]) if query_words else query.title()

                # Categorize sentences into thematic pillars for broad coverage
                def_points = [s for s in clean_matches if any(k in s.lower() for k in ["is ", "are ", "defined", "called", "subset", "refers to", "type of", "known as"])]
                mech_points = [s for s in clean_matches if any(k in s.lower() for k in ["use", "uses", "network", "system", "algorithm", "learn", "process", "function", "technique", "feature"])]
                hist_points = [s for s in clean_matches if any(k in s.lower() for k in ["research", "199", "200", "pioneer", "award", "led to", "history", "invent", "origin", "developed"])]

                lines = [
                    f"### 📖 {primary_subject} — Comprehensive Conceptual Analysis\n",
                    f"{clean_matches[0]}\n"
                ]

                # 1. Theoretical Framework & Core Definition
                if def_points:
                    lines.append("#### 🏛️ 1. Definition & Foundational Principles\n")
                    for p in def_points[:3]:
                        lines.append(f"* {p}\n")

                # 2. Mechanisms, Architecture & Operation
                remaining_mech = [p for p in mech_points if p not in def_points]
                if remaining_mech:
                    lines.append("#### ⚙️ 2. Architectural Structure & How It Operates\n")
                    for p in remaining_mech[:3]:
                        lines.append(f"* {p}\n")

                # 3. Research Milestones, Timeline & Context
                remaining_hist = [p for p in hist_points if p not in def_points and p not in mech_points]
                if remaining_hist:
                    lines.append("#### 📜 3. Milestones, Research & Practical Context\n")
                    for p in remaining_hist[:3]:
                        lines.append(f"* {p}\n")

                # Fallback for remaining unmatched key facts
                used_points = set(def_points + remaining_mech + remaining_hist + [clean_matches[0]])
                other_points = [p for p in clean_matches if p not in used_points]
                if other_points and len(lines) < 10:
                    lines.append("#### 📌 4. Key Notes & Core Takeaways\n")
                    for p in other_points[:3]:
                        lines.append(f"* {p}\n")

                return {"output": "\n".join(lines)}
            else:
                primary_subject = query.title()
                lines = [
                    f"### 📖 {primary_subject} — Conceptual Overview\n",
                    "Here is the foundational academic context extracted from your uploaded study material:\n"
                ]
                for s in raw_sentences[:5]:
                    cleaned = s.strip("-:;, •")
                    lines.append(f"* {cleaned}\n")
                return {"output": "\n".join(lines)}

        # ── 3. GENERAL FALLBACK (No document loaded) ──
        tool = self.tools_map["concept_explainer_tool"]
        res = tool.run({"concept": query, "mode": "Easy (Beginner-Friendly)", "context": context})
        return {"output": res}

class ResilientAgentWrapper:
    """Wraps primary AgentExecutor with FallbackStudyAgent for guaranteed zero-crash execution."""
    def __init__(self, primary_agent: Any, fallback_agent: FallbackStudyAgent):
        self.primary_agent = primary_agent
        self.fallback_agent = fallback_agent

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.primary_agent:
            try:
                res = self.primary_agent.invoke(inputs)
                if res and res.get("output") and len(str(res.get("output")).strip()) > 10:
                    return res
            except Exception as err:
                logger.warning(f"Primary Agent encountered error ({err}), falling back to deterministic tool router.")
        return self.fallback_agent.invoke(inputs)

def create_study_buddy_agent(
    gemini_key: Optional[str] = None,
    openai_key: Optional[str] = None
) -> Any:
    """
    Creates and returns a LangChain Tool Calling Agent.
    Wraps execution in ResilientAgentWrapper to guarantee fallback operation on API errors/quota limits.
    """
    tools = get_all_study_tools()
    fallback = FallbackStudyAgent(tools)
    
    active_gemini_key = gemini_key if gemini_key is not None else GEMINI_API_KEY
    active_openai_key = openai_key if openai_key is not None else OPENAI_API_KEY

    llm = None

    if active_gemini_key:
        try:
            from config import DEFAULT_GEMINI_MODEL
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_MODEL,
                google_api_key=active_gemini_key,
                temperature=0.3,
                max_retries=1,
                timeout=10
            )
            logger.info(f"Initialized {DEFAULT_GEMINI_MODEL} for Study Buddy Agent.")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGoogleGenerativeAI: {e}")

    if not llm and active_openai_key:
        try:
            from config import DEFAULT_OPENAI_MODEL
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=DEFAULT_OPENAI_MODEL,
                api_key=active_openai_key,
                temperature=0.3,
                max_retries=1,
                timeout=10
            )
            logger.info("Initialized ChatOpenAI for Study Buddy Agent.")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOpenAI: {e}")

    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT_TEMPLATE + "\n\nAvailable Context / Notes Content:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
            return ResilientAgentWrapper(executor, fallback)
        except Exception as err:
            logger.error(f"Error creating tool calling agent executor: {err}")

    logger.info("Using FallbackStudyAgent mode for reliable tool routing.")
    return fallback
