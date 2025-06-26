import streamlit as st
import threading
import uvicorn
import requests
from api import app as fastapi_app
import base64
import io
import streamlit.components.v1 as components
import time
import hashlib

# ---------- Utility: safe Streamlit rerun (supports old and new API) ----------

def _safe_rerun() -> None:
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    elif hasattr(st, "rerun"):
        st.rerun()

# ---------------------------------------------------------------------------
# Prefer streamlit_mic_recorder which supports custom start/stop prompts
# ---------------------------------------------------------------------------

# 1st choice: streamlit_mic_recorder (supports separate start/stop prompts)

def _get_mic_bytes() -> bytes | None:
    """Return WAV bytes from whichever recorder library is available."""
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore

        rec = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="🛑 Stop Recording",
            format="wav",
            key="mic_recorder",
        )
        return rec["bytes"] if rec else None

    except ModuleNotFoundError:
        # Fallback to audio_recorder_streamlit (single label, rely on icon color)
        try:
            from audio_recorder_streamlit import audio_recorder  # type: ignore

            return audio_recorder(
                text="🎙️ Start Recording",  # single text
                icon_name="microphone",
                icon_size="2x",
                neutral_color="#6aa36f",
                recording_color="#e8b62c",
                key="audio_rec_fallback",
            )
        except ModuleNotFoundError:
            return None

# Page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ── Basic styling tweaks ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* improve chat width */
    section.main > div { max-width: 900px; margin-left: auto; margin-right: auto; }
    /* bigger buttons */
    .stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Upload Document
st.sidebar.header("📄 Document Indexing")
st.sidebar.write("Upload any PDF or Word document. The content will be vector-indexed so the chatbot can answer questions about it.")

uploaded_files = st.sidebar.file_uploader("Select PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if st.sidebar.button("Index Document(s)"):
    if not uploaded_files:
        st.sidebar.warning("Please choose at least one file.")
    else:
        file_tuple = [
            ("files", (uf.name, uf, uf.type)) for uf in uploaded_files
        ]
        try:
            resp = requests.post("http://localhost:8005/upload/", files=file_tuple)
            if resp.ok:
                task_id = resp.json().get("task_id")
                st.session_state['messages'] = []
                progress_bar = st.sidebar.progress(0)
                progress_text = st.sidebar.empty()
                pct = 0
                while pct < 100 and pct >= 0:
                    try:
                        p_resp = requests.get(f"http://localhost:8005/progress/{task_id}")
                        if p_resp.ok:
                            pct = p_resp.json().get("progress", 0)
                            if pct < 0:
                                raise RuntimeError("Indexing failed")
                            progress_bar.progress(min(pct, 100)/100.0)
                            progress_text.text(f"Indexing progress: {pct}%")
                            time.sleep(0.5)
                        else:
                            break
                    except Exception:
                        break
                if pct == 100:
                    progress_text.text("Indexing progress: 100%")
                    st.sidebar.success("Indexing complete ✅")
                else:
                    progress_text.text(f"Indexing stopped at: {pct}%")
                    st.sidebar.error("Indexing did not complete.")
            else:
                st.sidebar.error(f"Error {resp.status_code}: {resp.json().get('detail')}")
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")

# Start FastAPI only once using session_state

def _start_api():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8005)

if 'api_started' not in st.session_state:
    threading.Thread(target=_start_api, daemon=True).start()
    st.session_state['api_started'] = True

# Main chat interface
st.title("📚 RAG Chatbot")
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'question_input' not in st.session_state:
    st.session_state['question_input'] = ""
if st.session_state.get('transcript_pending'):
    st.session_state['question_input'] = st.session_state.pop('transcript_pending')

# Display existing chat
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# --- Interactive input section --------------------------------------------------

st.divider()
st.subheader("Ask a question")

text_query = st.text_input("Type your question (optional)", key="question_input")

# --- Voice recording input ----------------------------------------------------

st.write("Or record your question:")

status_placeholder = st.empty()

# Attempt to use recorder; if unavailable, fall back to upload
audio_bytes = _get_mic_bytes()

# Check if we have new audio to transcribe
if audio_bytes is not None:
    # Create hash of audio to detect if it's new
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    last_audio_hash = st.session_state.get('last_audio_hash', '')
    
    # Only transcribe if this is new audio and we don't have a pending transcript
    if (audio_hash != last_audio_hash and 
        'transcript_pending' not in st.session_state and 
        not st.session_state.get('transcribing_in_progress', False)):
        
        st.session_state['transcribing_in_progress'] = True
        st.session_state['last_audio_hash'] = audio_hash
        
        try:
            with st.spinner("Transcribing audio..."):
                files = {"audio": ("question.wav", audio_bytes, "audio/wav")}
                tr_resp = requests.post("http://localhost:8005/transcribe/", files=files)
                if tr_resp.ok:
                    transcript = tr_resp.json().get("text", "")
                    st.session_state['transcript_pending'] = transcript
                    st.session_state['transcribing_in_progress'] = False
                    st.rerun()
        except Exception as e:
            st.session_state['transcribing_in_progress'] = False
            st.error(f"Transcription error: {e}")

# Reset transcription flag when transcript is ready
if st.session_state.get('transcript_pending'):
    st.session_state['transcribing_in_progress'] = False

# voice select addition near input area before send button
voice_choice = "nova"  # default TTS voice

if st.button("Send"):
    if not text_query.strip():
        st.warning("Please provide a question (text or record audio).")
        st.stop()

    # ensure we're only sending text; audio already transcribed
    if 'transcript_pending' in st.session_state or audio_bytes is None:
        audio_bytes = None

    audio_only = bool(audio_bytes and not text_query)
    with st.spinner("Thinking..."):
        try:
            if audio_bytes:
                files = {"audio": ("question.wav", audio_bytes, "audio/wav")}
                data = {"voice": voice_choice}
                if text_query:
                    data["question"] = text_query
                resp = requests.post("http://localhost:8005/ask/", files=files, data=data)
            else:
                resp = requests.post("http://localhost:8005/ask/", json={"question": text_query, "voice": voice_choice})

            if resp.ok:
                resp_json = resp.json()
                answer_text = resp_json.get("answer")
                answer_audio_b64 = resp_json.get("answer_audio", "")
                question_text = resp_json.get("question", text_query or '🔊 Audio question')
                # Render user message now (text or transcript from backend)
                final_user_text = question_text
                st.session_state['messages'].append({'role': 'user', 'content': final_user_text})
                st.chat_message('user').write(final_user_text)
            else:
                answer_text = f"Error {resp.status_code}: {resp.json().get('detail')}"
                answer_audio_b64 = ""
        except Exception as e:
            answer_text = f"Connection error: {e}"
            answer_audio_b64 = ""

    # Render assistant response
    st.session_state['messages'].append({'role': 'assistant', 'content': answer_text})
    st.chat_message('assistant').write(answer_text)

    # Play audio answer
    if answer_audio_b64 and answer_audio_b64.startswith("data:audio"):
        b64_data = answer_audio_b64.split(",", 1)[1]
        audio_data = base64.b64decode(b64_data)
        st.audio(audio_data, format='audio/mp3')

    # Clear audio buffer so user must record again next turn
    audio_bytes = None
    st.session_state.pop('transcript_pending', None)
    st.session_state['transcribing_in_progress'] = False
    st.session_state.pop('last_audio_hash', None)  # Reset audio hash for next recording
