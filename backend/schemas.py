# backend/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    session_context: Optional[str] = None

class ChatResponse(BaseModel):
    summary: str
    sql_query: str
    chart_type: str
    data: List[Dict[str, Any]]

class UploadResponse(BaseModel):
    filename: str
    message: str
    session_context: str