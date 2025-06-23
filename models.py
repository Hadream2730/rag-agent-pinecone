from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
