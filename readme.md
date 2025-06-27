# RAG Chatbot with Pinecone

A RAG chatbot that indexes documents in Pinecone and answers questions with text and audio responses.

## Setup

### 1. Clone and Environment
```bash
git clone https://github.com/Hadream2730/rag-agent-pinecone.git
cd rag-agent-pinecone
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=sk-your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-agent-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
API_BASE_URL=http://localhost:8005
```

## Running on Server

### Start API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8005
```

### Start UI (in another terminal)
```bash
streamlit run Chatbot_UI.py --server.port 8501 --server.address 0.0.0.0
```

### Access
- UI: `http://your-server-ip:8501`
- API: `http://your-server-ip:8005`

**Note**: Update `API_BASE_URL` in `.env` to your server's IP/domain if running UI and API on different servers.