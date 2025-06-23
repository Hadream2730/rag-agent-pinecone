import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Any
from models import UploadResponse, AskRequest, AskResponse
from embedding_creator import create_pinecone_index
from chatbot import answer_question

# ─── App & Directories ────────────────────────────────────
app = FastAPI()
UPLOAD_DIR = "uploads"

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

# ─── Upload Endpoint ─────────────────────────────────────
@app.post("/upload/", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Receive multiple .pdf or .docx files, save them, build a Pinecone index, 
    and store it on app.state.
    """
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

    # Index creation timing
    print("Creating Pinecone index...")
    index_start_time = time.time()
    app.state.pinecone_index = create_pinecone_index(paths)
    index_time = time.time() - index_start_time
    print(f"Pinecone index populated in {index_time:.2f} seconds for {len(files)} file(s)")
    
    total_upload_time = time.time() - upload_start_time
    print(f"Total upload processing time: {total_upload_time:.2f} seconds")
    print(f"Upload completed successfully at {time.strftime('%H:%M:%S')}")
    print("-" * 60)
    
    return UploadResponse(message=f"Indexed {len(files)} document(s)")

# ─── Dependency to fetch Pinecone index ─────────────────
def get_pinecone_index():
    idx = app.state.pinecone_index
    if idx is None:
        print("No Pinecone index available")
        raise HTTPException(400, "No index available. Upload first.")
    return idx

# ─── Ask Endpoint ────────────────────────────────────────
@app.post("/ask/", response_model=AskResponse)
async def ask(question_req: AskRequest, pinecone_index: Any = Depends(get_pinecone_index)):
    """
    Await the async answer_question, passing in the stored Pinecone index.
    """
    print(f"API request received at {time.strftime('%H:%M:%S')}")
    api_start_time = time.time()
    
    answer = await answer_question(pinecone_index, question_req.question)
    
    api_total_time = time.time() - api_start_time
    print(f"API response completed in {api_total_time:.2f} seconds")
    print("=" * 60)
    
    return AskResponse(answer=answer)




