from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)

class SourceChunk(BaseModel):
    doc_id: str
    source: str
    chunk: str
    score: float

class REvalResponse(BaseModel):
    recall: float 
    precision: float
    hit_rate: float 
    mrr: float


class REvalRequest(BaseModel):
    top_k: int = Field(5, ge=1, le=50)
    k: int = Field(5, ge=1, le=50)

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    index_size: int
    device: str