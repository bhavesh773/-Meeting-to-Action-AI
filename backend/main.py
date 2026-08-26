import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
from diarization import diarize_and_align, format_timestamp

app = FastAPI(
    title="Meeting-to-Action AI API (Next-Gen)",
    description="Speech-to-text, meeting intelligence, sentiment analysis, and interactive meeting Q&A API",
    version="3.0.0"
)

# Allow Angular frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory configuration
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Maximum file size (100 MB)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

# Supported audio/video extensions
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".webm", ".wma", ".mp4", ".mov", ".mkv"
}

# Global Whisper model
whisper_model = None

@app.on_event("startup")
def load_model():
    global whisper_model
    try:
        print("Loading Whisper model (base)...")
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to pre-load Whisper model: {e}")

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model on-demand...")
        whisper_model = whisper.load_model("base")
    return whisper_model


# --- NLP & Semantic Analysis Functions ---

def format_timestamp(seconds: float) -> str:
    """Formats float seconds into mm:ss string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def split_into_sentences(text: str) -> List[str]:
    """Splits text into clean sentences while respecting abbreviations and numbers."""
    if not text:
        return []

    abbreviations = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "Ltd.", "vs.", "e.g.", "i.e.", "approx.", "dept.", "est."]
    protected = text
    for abbr in abbreviations:
        protected = re.sub(r'\b' + re.escape(abbr), abbr.replace('.', '@DOT@'), protected, flags=re.IGNORECASE)

    protected = re.sub(r'(\d+)\.(\d+)', r'\1@DOT@\2', protected)
    raw_sentences = re.split(r'[\.\?\!]+(?:\s+|$)', protected)

    sentences = []
    for s in raw_sentences:
        clean = s.replace('@DOT@', '.').strip()
        if clean and len(clean) > 2:
            sentences.append(clean)
    return sentences if sentences else [s.strip() for s in text.split("\n") if s.strip()]


def generate_smart_summary(transcript: str, max_sentences: int = 4) -> str:
    """Generates a concise executive summary from the transcript."""
    if not transcript or not transcript.strip():
        return "No audible transcript content to summarize."

    sentences = split_into_sentences(transcript)
    if len(sentences) <= max_sentences:
        return ". ".join(sentences) + ("." if not sentences[-1].endswith(".") else "")

    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "i", "you", "he", "she",
        "it", "we", "they", "this", "that", "these", "those", "from", "as",
        "so", "my", "your", "his", "her", "its", "our", "their", "just", "like"
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
    freq: Dict[str, int] = {}
    for word in words:
        if word not in stopwords:
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return ". ".join(sentences[:max_sentences]) + "."

    max_freq = max(freq.values())
    normalized_freq = {k: v / max_freq for k, v in freq.items()}

    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        if not sent_words:
            continue
        score = sum(normalized_freq.get(w, 0) for w in sent_words) / (len(sent_words) ** 0.5)
        if idx == 0 or idx == len(sentences) - 1:
            score *= 1.2
        scored_sentences.append((score, idx, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])

    summary = ". ".join([s[2].rstrip(".") for s in top_sentences]) + "."
    return summary


def extract_key_points(transcript: str) -> List[str]:
    """Extracts high-level key takeaways and important discussion points."""
    sentences = split_into_sentences(transcript)
    if not sentences:
        return ["Meeting recording processed successfully."]

    key_points = []
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(indicator in s_lower for indicator in [
            "discussed", "highlighted", "focus on", "goal is", "important",
            "feature", "milestone", "plan is", "agreed", "finalized", "roadmap", "architecture", "timeline"
        ]):
            key_points.append(sentence.rstrip("."))

    if len(key_points) < 2:
        for s in sentences[:4]:
            if s.rstrip(".") not in key_points:
                key_points.append(s.rstrip("."))

    return key_points[:6]


def extract_decisions(transcript: str) -> List[Dict[str, str]]:
    """Extracts decisions agreed upon during the meeting."""
    sentences = split_into_sentences(transcript)
    decisions = []
    seen = set()

    decision_patterns = [
        r'\b(?:decided|agreed|finalized|chosen|selected)\s+(?:to\s+|that\s+|on\s+)?(.+)',
        r'\bwe will\s+(?:use|deploy|build|go with|implement)\s+(.+)',
        r'\b(?:decision is|the plan is)\s+(?:to\s+|that\s+)?(.+)',
        r'\bthe deadline\s+(?:will be|is set to)\s+(.+)'
    ]

    for sentence in sentences:
        s_clean = sentence.strip().rstrip(".")
        s_lower = s_clean.lower()

        for pattern in decision_patterns:
            match = re.search(pattern, s_clean, re.IGNORECASE)
            if match:
                details = match.group(1).strip()
                topic = "General Decision"
                if "frontend" in s_lower or "angular" in s_lower or "ui" in s_lower or "react" in s_lower:
                    topic = "Frontend Architecture"
                elif "backend" in s_lower or "fastapi" in s_lower or "api" in s_lower or "python" in s_lower:
                    topic = "Backend Architecture"
                elif "deploy" in s_lower or "vercel" in s_lower or "aws" in s_lower or "cloud" in s_lower:
                    topic = "Deployment & Hosting"
                elif "database" in s_lower or "mongodb" in s_lower or "sql" in s_lower:
                    topic = "Database"
                elif "deadline" in s_lower or "friday" in s_lower or "monday" in s_lower or "august" in s_lower:
                    topic = "Project Timeline"
                elif "test" in s_lower or "qa" in s_lower:
                    topic = "Quality & Testing"

                if details.lower() not in seen:
                    seen.add(details.lower())
                    decisions.append({
                        "topic": topic,
                        "details": s_clean
                    })
                break

    if not decisions and len(sentences) >= 2:
        for sentence in sentences:
            if any(w in sentence.lower() for w in ["will", "should", "plan", "scheduled"]):
                decisions.append({
                    "topic": "Agreed Approach",
                    "details": sentence.rstrip(".")
                })
                if len(decisions) >= 3:
                    break

    return decisions[:6]


def extract_action_items(transcript: str) -> List[Dict[str, str]]:
    """Extracts action items with tasks, assignees, deadlines, and priorities."""
    if not transcript:
        return []

    sentences = split_into_sentences(transcript)
    action_keywords = [
        "will", "should", "need to", "must", "assign", "complete", "send",
        "prepare", "follow up", "schedule", "implement", "review", "update",
        "create", "fix", "deliver", "organize", "ensure", "check", "share",
        "coordinate", "reach out", "finish", "test"
    ]

    action_items = []
    seen = set()

    for sentence in sentences:
        clean_sentence = sentence.strip().rstrip(".")
        lower_sentence = clean_sentence.lower()

        matched_keywords = [kw for kw in action_keywords if re.search(r'\b' + re.escape(kw) + r'\b', lower_sentence)]
        if not matched_keywords:
            continue

        if lower_sentence in seen:
            continue
        seen.add(lower_sentence)

        # Detect Assignee
        assignee = "Team"
        assignee_match = re.search(r'\b([A-Z][a-z]+)\s+(?:will|should|must|needs? to|is assigned to)\b', clean_sentence)
        if assignee_match:
            assignee = assignee_match.group(1)
        elif "i will" in lower_sentence or "i'll" in lower_sentence or "i am going to" in lower_sentence:
            assignee = "Speaker"
        elif "we will" in lower_sentence or "let's" in lower_sentence or "we need to" in lower_sentence:
            assignee = "Team"

        # Detect Deadline
        deadline = "Not specified"
        deadline_match = re.search(
            r'\b(?:by|before|on|due|until)\s+([A-Za-z0-9\s]+?)(?:[\.,]|$|\s+for|\s+and|\s+so|\s+before)',
            clean_sentence,
            re.IGNORECASE
        )
        if deadline_match:
            cand = deadline_match.group(1).strip()
            if len(cand) > 1 and len(cand) < 25:
                deadline = cand.capitalize()
        elif "tomorrow" in lower_sentence:
            deadline = "Tomorrow"
        elif "today" in lower_sentence:
            deadline = "Today"
        elif "next week" in lower_sentence:
            deadline = "Next Week"
        elif "friday" in lower_sentence:
            deadline = "Friday"
        elif "monday" in lower_sentence:
            deadline = "Monday"

        # Detect Priority
        priority = "Medium"
        if any(w in lower_sentence for w in ["urgent", "asap", "immediately", "must", "critical", "today", "high priority", "blocker"]):
            priority = "High"
        elif any(w in lower_sentence for w in ["when possible", "eventually", "later", "low priority", "if time permits", "nice to have"]):
            priority = "Low"

        action_items.append({
            "task": clean_sentence,
            "assignee": assignee,
            "deadline": deadline,
            "priority": priority
        })

    return action_items[:8]


def analyze_sentiment_and_tone(transcript: str) -> Dict[str, Any]:
    """Analyzes overall meeting sentiment, tone, and conversational mood."""
    if not transcript:
        return {
            "overall": "Neutral",
            "score": 50,
            "tone": "Informative",
            "topics": ["General Discussion"]
        }

    lower = transcript.lower()

    positive_words = ["great", "good", "excellent", "progress", "milestone", "success", "achieved", "completed", "positive", "growth", "approved", "love", "awesome"]
    concern_words = ["urgent", "issue", "bug", "blocker", "delay", "problem", "risk", "critical", "warning", "difficult", "fail", "hard"]
    action_words = ["deadline", "deliverable", "assign", "schedule", "implement", "release", "ship", "launch", "prepare"]

    pos_count = sum(1 for w in positive_words if w in lower)
    concern_count = sum(1 for w in concern_words if w in lower)
    action_count = sum(1 for w in action_words if w in lower)

    score = 50 + (pos_count * 8) - (concern_count * 8)
    score = max(10, min(95, score))

    if score > 65:
        overall = "Positive & Productive"
        tone = "Optimistic & Forward-Looking"
    elif score < 40:
        overall = "Action-Oriented (High Urgency)"
        tone = "Critical & Focus-Driven"
    else:
        overall = "Balanced & Constructive"
        tone = "Collaborative & Structured"

    # Extract dynamic discussion topics
    topics = []
    if any(w in lower for w in ["frontend", "ui", "ux", "angular", "design"]):
        topics.append("Frontend & UI")
    if any(w in lower for w in ["backend", "api", "fastapi", "database", "server"]):
        topics.append("Backend & Architecture")
    if any(w in lower for w in ["test", "testing", "qa", "audit"]):
        topics.append("Testing & Quality")
    if any(w in lower for w in ["deploy", "vercel", "cloud", "release"]):
        topics.append("Deployment & DevOps")
    if any(w in lower for w in ["timeline", "deadline", "schedule", "roadmap"]):
        topics.append("Timeline & Milestones")

    if not topics:
        topics = ["Project Strategy", "Team Collaboration"]

    return {
        "overall": overall,
        "score": score,
        "tone": tone,
        "topics": topics
    }


def safe_transcribe(file_path: str, expected_speakers: Optional[int] = None) -> Dict[str, Any]:
    """Safely loads, decodes, transcribes audio with Whisper, and performs acoustic speaker diarization."""
    try:
        audio = whisper.load_audio(file_path)
    except Exception as decode_err:
        print(f"Audio decoding error: {decode_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded audio file. Please ensure it is a valid, uncorrupted audio recording."
        )

    if audio is None or len(audio) == 0:
        return {
            "text": "No audible audio stream found in the uploaded file.",
            "duration": 0.0,
            "segments": [],
            "num_speakers": 0,
            "speaker_stats": []
        }

    duration_seconds = len(audio) / 16000.0

    # Pad short audio (< 1s) with zeros
    if len(audio) < 16000:
        import numpy as np
        audio = np.pad(audio, (0, 16000 - len(audio)))

    model = get_whisper_model()
    try:
        result = model.transcribe(audio, fp16=False)
        text = result.get("text", "").strip()
        raw_segments = result.get("segments", [])

        # Run acoustic speaker diarization and alignment
        diarized = diarize_and_align(
            audio=audio,
            sample_rate=16000,
            raw_segments=raw_segments,
            expected_speakers=expected_speakers
        )

        return {
            "text": text if text else "No audible speech was detected in the provided file.",
            "duration": duration_seconds,
            "segments": diarized["segments"],
            "num_speakers": diarized["num_speakers"],
            "speaker_stats": diarized["speaker_stats"]
        }
    except Exception as transcribe_err:
        err_str = str(transcribe_err).lower()
        if "cannot reshape tensor of 0 elements" in err_str or "0 elements" in err_str:
            return {
                "text": "No audible speech was detected in the provided file.",
                "duration": duration_seconds,
                "segments": [],
                "num_speakers": 0,
                "speaker_stats": []
            }
        raise transcribe_err


# --- API Models ---

class SpeakerColorThemeModel(BaseModel):
    bg: str
    border: str
    text: str
    name: str

class SpeakerStatModel(BaseModel):
    speaker: str
    speaker_index: int
    talk_time_seconds: float
    talk_time_formatted: str
    talk_time_percentage: float
    word_count: int
    segments_count: int
    color_theme: SpeakerColorThemeModel

class ActionItemModel(BaseModel):
    task: str
    assignee: str
    deadline: str
    priority: str

class DecisionModel(BaseModel):
    topic: str
    details: str

class SegmentModel(BaseModel):
    timestamp: str
    start: float
    end: float
    speaker: str
    speaker_id: Optional[int] = 1
    text: str

class InsightsModel(BaseModel):
    duration_formatted: str
    duration_seconds: float
    word_count: int
    action_items_count: int
    decisions_count: int
    num_speakers: int

class SentimentModel(BaseModel):
    overall: str
    score: int
    tone: str
    topics: List[str]

class ProcessResponse(BaseModel):
    success: bool
    filename: str
    transcript: str
    summary: str
    key_points: List[str]
    decisions: List[DecisionModel]
    action_items: List[str]
    structured_action_items: List[ActionItemModel]
    segments: List[SegmentModel]
    num_speakers: int
    speaker_stats: List[SpeakerStatModel]
    insights: InsightsModel
    sentiment: SentimentModel
    error: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    transcript: str
    summary: Optional[str] = ""
    action_items: Optional[List[Dict[str, Any]]] = []
    decisions: Optional[List[Dict[str, Any]]] = []
    speaker_stats: Optional[List[Dict[str, Any]]] = []

class ChatResponse(BaseModel):
    answer: str
    suggested_followups: List[str]


# --- Routes ---

@app.get("/")
def home():
    return {
        "message": "Meeting-to-Action AI Backend Running 🚀",
        "status": "online",
        "version": "3.1.0",
        "endpoints": {
            "health": "/",
            "process": "/process (POST)",
            "chat": "/chat (POST)"
        }
    }


@app.post("/process", response_model=ProcessResponse)
async def process_meeting(
    file: UploadFile = File(...),
    expected_speakers: Optional[int] = Form(None)
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename is missing."
        )

    file_ext = Path(file.filename).suffix.lower()
    # Support direct live recordings without explicit extension
    if not file_ext:
        file_ext = ".webm"

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    unique_filename = f"{uuid.uuid4().hex}_{Path(file.filename).name or 'recording'}{file_ext if not Path(file.filename).suffix else ''}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        contents = await file.read()
        file_size = len(contents)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            )

        with open(file_path, "wb") as f:
            f.write(contents)

        transcribe_data = await asyncio.to_thread(safe_transcribe, file_path, expected_speakers)
        transcript = transcribe_data["text"]
        duration_sec = transcribe_data["duration"]
        segments = transcribe_data["segments"]
        num_speakers = transcribe_data["num_speakers"]
        speaker_stats = transcribe_data["speaker_stats"]

        # Information Extraction Pipeline
        summary = generate_smart_summary(transcript)
        key_points = extract_key_points(transcript)
        decisions = extract_decisions(transcript)
        structured_actions = extract_action_items(transcript)
        sentiment_data = analyze_sentiment_and_tone(transcript)

        string_actions = [
            f"{item['task']} [{item['assignee']} • Deadline: {item['deadline']} • {item['priority']} Priority]"
            for item in structured_actions
        ]

        words = len(transcript.split()) if transcript else 0
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        duration_fmt = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        insights = InsightsModel(
            duration_formatted=duration_fmt,
            duration_seconds=round(duration_sec, 1),
            word_count=words,
            action_items_count=len(structured_actions),
            decisions_count=len(decisions),
            num_speakers=num_speakers
        )

        return ProcessResponse(
            success=True,
            filename=file.filename,
            transcript=transcript,
            summary=summary,
            key_points=key_points,
            decisions=[DecisionModel(**d) for d in decisions],
            action_items=string_actions,
            structured_action_items=[ActionItemModel(**item) for item in structured_actions],
            segments=[SegmentModel(**seg) for seg in segments],
            num_speakers=num_speakers,
            speaker_stats=[SpeakerStatModel(**stat) for stat in speaker_stats],
            insights=insights,
            sentiment=SentimentModel(**sentiment_data)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing meeting audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process meeting audio: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_err:
                print(f"Warning: Could not remove temporary file {file_path}: {cleanup_err}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_meeting(req: ChatRequest):
    """Answers user questions regarding the meeting transcript using contextual AI reasoning."""
    q = req.question.strip().lower()
    transcript = req.transcript.strip()

    if not transcript:
        return ChatResponse(
            answer="I don't have any transcript data for this meeting yet. Please process a meeting recording first!",
            suggested_followups=["Upload an audio file", "Record a meeting"]
        )

    sentences = split_into_sentences(transcript)

    # 1. Questions about Speakers / Participants / Talk-Time
    if any(w in q for w in ["speaker", "who spoke", "participants", "how many people", "talk time", "who talked the most"]):
        if req.speaker_stats and len(req.speaker_stats) > 0:
            total_spk = len(req.speaker_stats)
            ans = f"There were **{total_spk} distinct speaker{'s' if total_spk > 1 else ''}** identified in this meeting:\n\n"
            for stat in req.speaker_stats:
                ans += f"• **{stat.get('speaker')}**: {stat.get('talk_time_formatted')} ({stat.get('talk_time_percentage')}%) — {stat.get('word_count')} words across {stat.get('segments_count')} turns\n"
            return ChatResponse(
                answer=ans.strip(),
                suggested_followups=["What were the key takeaways?", "Who is assigned to each action item?", "Draft a recap email"]
            )

    # 2. Questions about Deadlines / Schedule / Dates
    if any(w in q for w in ["deadline", "due", "when", "date", "schedule", "time", "friday", "monday"]):
        deadline_matches = [s for s in sentences if any(w in s.lower() for w in ["deadline", "by", "before", "due", "friday", "monday", "august", "tomorrow", "next week"])]
        if deadline_matches:
            ans = "Here are the deadlines and timelines mentioned in the meeting:\n\n"
            for s in deadline_matches[:4]:
                ans += f"• {s}\n"
            return ChatResponse(
                answer=ans.strip(),
                suggested_followups=["Who is assigned to each task?", "What are the key decisions?", "Give me a 2-sentence summary"]
            )

    # 3. Questions about Tasks / Action Items / Responsibilities
    if any(w in q for w in ["task", "action", "responsibility", "assigned", "who will", "who is doing", "what needs to be done"]):
        if req.action_items and len(req.action_items) > 0:
            ans = "Here are the action items and assigned responsibilities:\n\n"
            for i, item in enumerate(req.action_items, 1):
                ans += f"{i}. **{item.get('task')}** (Assigned: *{item.get('assignee', 'Team')}*, Due: *{item.get('deadline', 'Not specified')}*)\n"
            return ChatResponse(
                answer=ans.strip(),
                suggested_followups=["What are the highest priority items?", "Draft a follow-up email for the team", "What decisions were agreed on?"]
            )

    # 4. Questions about Decisions
    if any(w in q for w in ["decision", "decide", "agreed", "choice", "concluded"]):
        if req.decisions and len(req.decisions) > 0:
            ans = "Here are the key decisions finalized during the discussion:\n\n"
            for d in req.decisions:
                ans += f"• **{d.get('topic', 'Decision')}**: {d.get('details')}\n"
            return ChatResponse(
                answer=ans.strip(),
                suggested_followups=["What are the next steps?", "Who is responsible for the frontend?", "Summarize the meeting"]
            )

    # 5. Draft Email / Slack Update
    if any(w in q for w in ["email", "slack", "message", "draft", "write a follow up"]):
        ans = "Here is a drafted follow-up email ready to send:\n\n"
        ans += "**Subject:** Meeting Recap & Next Steps\n\n"
        ans += "Hi Team,\n\nHere is a quick summary of our recent discussion:\n\n"
        ans += f"{req.summary or transcript[:200]}...\n\n"
        if req.action_items:
            ans += "**Action Items:**\n"
            for item in req.action_items[:4]:
                ans += f"- {item.get('task')} ({item.get('assignee')}, Due: {item.get('deadline')})\n"
        ans += "\nPlease let me know if anyone has questions.\n\nBest regards,\nTeam"
        return ChatResponse(
            answer=ans,
            suggested_followups=["List all decisions", "What are the deadlines?", "Show the full transcript"]
        )

    # 6. Semantic keyword search match in sentences
    words = [w for w in re.findall(r'\b\w{3,}\b', q) if w not in ["what", "when", "where", "who", "which", "how", "the", "and", "about", "tell", "show"]]
    relevant = []
    for s in sentences:
        s_lower = s.lower()
        if any(w in s_lower for w in words):
            relevant.append(s)

    if relevant:
        ans = f"Based on the meeting discussion:\n\n"
        for s in relevant[:4]:
            ans += f"• {s}\n"
        return ChatResponse(
            answer=ans.strip(),
            suggested_followups=["What are the action items?", "What decisions were made?", "Draft a follow-up email"]
        )

    # Default fallback answer using summary
    ans = f"Regarding your question, here is what was discussed in the meeting:\n\n{req.summary or transcript[:250]}."
    return ChatResponse(
        answer=ans,
        suggested_followups=["What are the deadlines?", "List all action items", "Draft a recap email"]
    )