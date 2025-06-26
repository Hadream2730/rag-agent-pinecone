import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request, BackgroundTasks
from fastapi.responses import ORJSONResponse
from typing import Any, Union
from models import UploadResponse, AskResponse
from embedding_creator import create_pinecone_index
from chatbot import answer_question
from audio_utils import transcribe_audio, synthesize_speech
import uuid

# ─── App & Directories ────────────────────────────────────
app = FastAPI(default_response_class=ORJSONResponse)
UPLOAD_DIR = "uploads"

# Global progress dictionary {task_id: percentage}
TASK_PROGRESS: dict[str, int] = {}

# Utility: remove a directory entirely
def _remove_directory(path: str):
    """Delete the directory and all its contents if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)

# Ensure directories are removed on app startup
@app.on_event("startup")
async def _startup_cleanup():
    print("Purging uploads directory on startup …")
    _remove_directory(UPLOAD_DIR)
    print("Startup purge completed.")

app.state.pinecone_index: Any | None = None

# Background task to build index and update progress
def _index_task(paths: list[str], task_id: str):
    TASK_PROGRESS[task_id] = 0

    def _cb(pct: int):
        TASK_PROGRESS[task_id] = pct

    try:
        app.state.pinecone_index = create_pinecone_index(paths, progress_cb=_cb)
        TASK_PROGRESS[task_id] = 100
    except Exception as e:
        TASK_PROGRESS[task_id] = -1  # indicates failure
        print(f"[Upload] indexing task {task_id} failed: {e}")

# ─── Upload Endpoint ─────────────────────────────────────
@app.post("/upload/", response_model=UploadResponse)
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
 
    # Clean previous uploads and index to free memory
    _remove_directory(UPLOAD_DIR)
    app.state.pinecone_index = None
    
    filenames = [f.filename for f in files]
    print(f"Upload started for {len(files)} file(s): {filenames} at {time.strftime('%H:%M:%S')}")
    upload_start_time = time.time()
    
    # Validate types and save files
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in {".pdf", ".docx"}:
            print(f"Invalid file type: {ext}")
            raise HTTPException(400, "Only .pdf and .docx supported")

    # File saving timing
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    save_start_time = time.time()
    paths = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    save_time = time.time() - save_start_time
    print(f"All files saved in {save_time:.2f} seconds")

    # Kick off background indexing task
    task_id = str(uuid.uuid4())
    background_tasks.add_task(_index_task, paths, task_id)

    print(f"Indexing started in background task {task_id}")

    return UploadResponse(task_id=task_id, message="Indexing started")

# ─── Dependency to fetch Pinecone index ─────────────────
def get_pinecone_index():
    idx = app.state.pinecone_index
    if idx is None:
        print("No Pinecone index available")
        raise HTTPException(400, "No index available. Upload first.")
    return idx

# ─── Ask Endpoint ────────────────────────────────────────
@app.post("/ask/", response_model=AskResponse)
async def ask(
    request: Request,
    question: str | None = Form(default=None),
    audio: Union[UploadFile, str, None] = File(default=None),
    voice: str = Form(default="alloy"),
    pinecone_index: Any = Depends(get_pinecone_index),
):
    """Handle text or audio question and return both text and audio answer."""
    print(f"API request received at {time.strftime('%H:%M:%S')}")
    api_start_time = time.time()

    # ------------------------------------------------------------------
    # Normalise input -> text
    # Accept either multipart/form-data (question &/or audio) or
    # raw JSON {"question": "..."}
    # ------------------------------------------------------------------

    if question is None and audio is None:
        # Try to parse JSON body for {"question": ...}
        try:
            data = await request.json()
            question = data.get("question") if isinstance(data, dict) else None
            # ignore `audio` field in JSON; audio upload must be multipart
        except Exception:
            question = None

    # Handle case where Swagger sends empty string for audio
    if isinstance(audio, str) and audio == "":
        audio = None

    # Treat placeholder or blank strings as empty
    if isinstance(question, str):
        if question.strip().lower() == "string" or question.strip() == "":
            question = None

    if question is None and audio is None:
        raise HTTPException(400, "Provide either 'question' text or 'audio' file")

    if question is None:
        question = await transcribe_audio(audio)
        print(f"[API] Transcribed audio question: {question}")

    # Log final question (for both text and audio cases)
    print(f"[API] Final question sent to Pinecone & LLM: {question}")

    # Log received audio file details
    if isinstance(audio, UploadFile):
        print(f"[API] Received audio file: {audio.filename} ({audio.content_type})")

    # RAG answer
    answer_text = await answer_question(pinecone_index, question)

    # TTS
    answer_audio_b64 = await synthesize_speech(answer_text, voice=voice)

    api_total_time = time.time() - api_start_time
    print(f"API response completed in {api_total_time:.2f} seconds")
    print("=" * 60)

    return AskResponse(question=question, answer=answer_text, answer_audio=answer_audio_b64)

# ─── Progress Endpoint ─────────────────────────────────────
@app.get("/progress/{task_id}")
async def progress(task_id: str):
    pct = TASK_PROGRESS.get(task_id)
    if pct is None:
        raise HTTPException(404, "Unknown task id")
    return {"progress": pct}

# ─── Transcribe Endpoint ─────────────────────────────────────
@app.post("/transcribe/")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    text = await transcribe_audio(audio)
    return {"text": text}




