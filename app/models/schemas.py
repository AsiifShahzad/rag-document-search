from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    session_id: str