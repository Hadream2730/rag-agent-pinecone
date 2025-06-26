import streamlit as st
import threading
import uvicorn
import requests
from api import app as fastapi_app
import base64
import io

# Optional audio recorder components (try st_audiorec first, then streamlit-mic-recorder)
AUDIOREC_FN = None
try:
    from audio_recorder_streamlit import audio_recorder  # type: ignore

    def _wrapper_ars():
        return audio_recorder("Start Recording", "Stop", icon_name="microphone", icon_size="2x", recording_color="#e8b62c", neutral_color="#6aa36f")

    AUDIOREC_FN = _wrapper_ars
except ModuleNotFoundError:
    try:
        from st_audiorec import st_audiorec  # type: ignore
        AUDIOREC_FN = st_audiorec
    except ModuleNotFoundError:
        try:
            from mic_recorder_streamlit import mic_recorder  # type: ignore

            def _wrapper_mic():
                return mic_recorder(start_prompt="Start recording", stop_prompt="Stop", show_text=False)

            AUDIOREC_FN = _wrapper_mic
        except ModuleNotFoundError:
            AUDIOREC_FN = None

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
        with st.spinner("Indexing documents..."):
            try:
                resp = requests.post("http://localhost:8005/upload/", files=file_tuple)
                if resp.ok:
                    # Clear previous chat
                    st.session_state['messages'] = []
                    st.sidebar.success(resp.json().get("message"))
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

# Display existing chat
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# --- Interactive input section --------------------------------------------------

st.divider()
st.subheader("Ask a question")

text_query = st.text_input("Type your question (optional)")

audio_bytes: bytes | None = None

if AUDIOREC_FN is not None:
    st.write("Or record your question:")
    audio_bytes = AUDIOREC_FN()
else:
    audio_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "opus"])
    if audio_file is not None:
        audio_bytes = audio_file.read()

# voice select addition near input area before send button
voice_choice = "nova"  # default TTS voice

if st.button("Send"):
    if not text_query and not audio_bytes:
        st.warning("Please provide text or record/upload audio.")
        st.stop()

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
