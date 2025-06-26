from pydantic import BaseModel

class UploadResponse(BaseModel):
    task_id: str
    message: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    answer_audio: str
