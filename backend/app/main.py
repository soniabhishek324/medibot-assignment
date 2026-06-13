from __future__ import annotations

import base64
import json
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.llm import generate_with_llm, local_document_answer
from app.models import ChatRequest, ChatResponse, HealthResponse, LoginRequest, LoginResponse, Source
from app.rbac import DEMO_USERS, UserSession, collections_for_role, rbac_refusal
from app.retrieval import get_retriever
from app.sql_rag import is_analytical_question, sql_rag_chain


app = FastAPI(title="MediBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_token(username: str, role: str, name: str) -> str:
    payload = json.dumps({"username": username, "role": role, "name": name}).encode()
    return base64.urlsafe_b64encode(payload).decode()


def decode_token(token: str) -> UserSession:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        return UserSession(username=payload["username"], name=payload["name"], role=payload["role"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid session token") from exc


def current_user(authorization: Annotated[str | None, Header()] = None) -> UserSession:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return decode_token(authorization.removeprefix("Bearer ").strip())


def requested_restricted_collection(question: str, role: str) -> str | None:
    lowered = question.lower()
    collection_terms = {
        "billing": ["billing", "claim", "insurance", "icd", "cpt"],
        "clinical": ["clinical", "diagnostic", "diagnosis", "drug", "formulary", "treatment"],
        "nursing": ["nursing", "icu", "infection", "patient care"],
        "equipment": ["equipment", "calibration", "maintenance", "fault code", "manual"],
    }
    allowed = set(collections_for_role(role))
    for collection, terms in collection_terms.items():
        if collection not in allowed and any(term in lowered for term in terms):
            return collection
    return None


@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    user = DEMO_USERS.get(payload.username)
    if not user or user["password"] != payload.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    role = user["role"]
    return LoginResponse(
        token=encode_token(payload.username, role, user["name"]),
        username=payload.username,
        display_name=user["name"],
        role=role,
        collections=collections_for_role(role),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: Annotated[UserSession, Depends(current_user)]) -> ChatResponse:
    restricted = requested_restricted_collection(payload.question, user.role)
    if restricted:
        return ChatResponse(answer=rbac_refusal(user.role, restricted), sources=[], retrieval_type="hybrid_rag", role=user.role, blocked=True)

    if is_analytical_question(payload.question):
        if user.role not in {"billing_executive", "admin"}:
            return ChatResponse(answer=rbac_refusal(user.role, "analytical SQL data"), sources=[], retrieval_type="sql_rag", role=user.role, blocked=True)
        return ChatResponse(answer=sql_rag_chain(payload.question), sources=[], retrieval_type="sql_rag", role=user.role)

    retrieved = get_retriever().retrieve(payload.question, user.role)
    contexts = [item.chunk.text for item in retrieved]
    system = "You are MediBot. Answer only from the supplied context and cite the provided sources implicitly."
    prompt = f"Question: {payload.question}\n\nContext:\n" + "\n\n---\n\n".join(contexts)
    answer = generate_with_llm(system, prompt) or local_document_answer(payload.question, contexts)
    sources = [
        Source(
            source_document=item.chunk.source_document,
            section_title=item.chunk.section_title,
            collection=item.chunk.collection,
        )
        for item in retrieved
    ]
    return ChatResponse(answer=answer, sources=sources, retrieval_type="hybrid_rag", role=user.role)


@app.get("/collections/{role}")
def collections(role: str) -> dict:
    return {"role": role, "collections": collections_for_role(role)}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=get_retriever().indexed_count)
