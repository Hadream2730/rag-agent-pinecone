from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    answer_audio: str
