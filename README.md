# 🎙️ Meeting-to-Action AI

An intelligent meeting assistant web application that transcribes audio recordings with OpenAI Whisper, generates smart summaries, and extracts actionable tasks with assignees and priority levels.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [OpenAI Whisper](https://github.com/openai/whisper), [PyTorch](https://pytorch.org/), [Uvicorn](https://www.uvicorn.org/)
- **Frontend**: [Angular 22](https://angular.dev/) (Standalone Components, Signals, Modern Control Flow), TypeScript, SCSS
- **Audio Processing**: [FFmpeg](https://ffmpeg.org/) (required by Whisper)

---

## 📋 Prerequisites

1. **Python**: Python 3.10 or higher
2. **Node.js**: Node 18+ and npm 9+
3. **FFmpeg**: Required for audio decoding:
   - **Windows**: `winget install Gyan.FFmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

The backend will be live at `http://127.0.0.1:8000` (API docs at `http://127.0.0.1:8000/docs`).

---

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install node dependencies
npm install

# Start the Angular development server
npm start
```

The frontend application will be live at `http://localhost:4200`.

---

## 🧪 Testing

```bash
cd frontend
npm test
```

---

## 🔒 Features & Improvements

- **Non-blocking Transcription**: Audio transcription runs in a dedicated thread pool via `asyncio.to_thread` to maintain high concurrency.
- **Secure File Isolation**: Uploaded files are assigned isolated UUID names, validated against format & size rules, and automatically cleaned up from disk.
- **Smart AI Summaries**: Extractive NLP scoring summarizes key discussion points without crude character truncation.
- **Structured Action Items**: Identifies tasks, detects assignees (Speaker, Team, or named participants), and calculates priority levels (`High`, `Medium`, `Low`).
- **Reactive Angular UI**: Uses Angular Signals, drag-and-drop file upload, live processing states, and one-click clipboard copying.
