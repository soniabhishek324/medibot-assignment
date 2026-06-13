from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    display_name: str
    role: str
    collections: list[str]


class ChatRequest(BaseModel):
    question: str = Field(min_length=2)


class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str
    blocked: bool = False


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
