from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import (
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.services.llm_service import LLMConfigurationError, LLMService
from backend.services.memory_service import ConversationMemory, ConversationNotFoundError
from backend.services.conversation_service import ConversationService
from backend.auth.google_oauth import GoogleOAuth, GoogleOAuthError
from backend.tools.base import ToolError
from backend.tools.registry import build_registry
from backend.agents.runtime import AgentRuntimeFactory, RuntimeConfigurationError
from backend.documents.errors import (
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.documents.ingestion import DocumentIngestionService
from backend.documents.models import (
    DocumentListResponse,
    DocumentManagementResponse,
    DocumentUploadResponse,
)
from backend.rag.errors import RAGFoundationError
from backend.rag.retrieval import RetrievalService

app = FastAPI(
    title="AI Assistant API",
    description="AI Assistant API with extensible tool calling.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST"],
    allow_headers=["Content-Type"],
)

conversation_memory = ConversationMemory(max_messages=20)
conversation_service: ConversationService | None = None
document_ingestion = DocumentIngestionService()
runtime_factory = AgentRuntimeFactory(
    retrieval_service=RetrievalService(
        document_ingestion.embedding_service,
        document_ingestion.vector_store,
    )
)
google_oauth = GoogleOAuth()
oauth_states: set[str] = set()


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None, description="The conversation to continue."
    )
    user_id: str = Field(default="anonymous", description="Trusted application user scope.")
    message: str = Field(..., description="The message to send to the assistant.")

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        conversation_id = value.strip()
        if not conversation_id:
            raise ValueError("conversation_id must not be empty.")
        return conversation_id

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be empty.")
        return message

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        user_id = value.strip()
        if not user_id:
            raise ValueError("user_id must not be empty.")
        return user_id


class ChatSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    page_number: int | None = None
    chunk_id: str


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    sources: list[ChatSource] | None = None


class HealthResponse(BaseModel):
    status: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request.",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    user_id: str = Form("anonymous"),
) -> DocumentUploadResponse:
    conversation_id = conversation_id.strip()
    user_id = user_id.strip() or "anonymous"
    if not conversation_id:
        raise HTTPException(status_code=422, detail="conversation_id must not be empty.")
    if not conversation_memory.has_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    try:
        metadata = document_ingestion.ingest(
            user_id=user_id,
            conversation_id=conversation_id,
            filename=file.filename or "",
            content_type=file.content_type,
            content=file.file,
        )
        return DocumentUploadResponse(
            document_id=metadata.document_id,
            filename=metadata.filename,
            content_type=metadata.content_type,
            status=metadata.status.value,
            conversation_id=metadata.conversation_id,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except (DocumentTooLargeError, InvalidDocumentError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RAGFoundationError as exc:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        ) from exc


def _document_scope(
    conversation_id: str, user_id: str
) -> tuple[str, str]:
    conversation = conversation_id.strip()
    user = user_id.strip() or "anonymous"
    if not conversation:
        raise HTTPException(
            status_code=422, detail="conversation_id must not be empty."
        )
    if not conversation_memory.has_conversation(conversation):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return conversation, user


def _document_response(
    metadata: Any, chunk_count: int
) -> DocumentManagementResponse:
    return DocumentManagementResponse(
        document_id=metadata.document_id,
        filename=metadata.filename,
        content_type=metadata.content_type,
        status=metadata.status.value,
        conversation_id=metadata.conversation_id,
        user_id=metadata.user_id,
        chunk_count=chunk_count,
    )


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    conversation_id: str,
    user_id: str = "anonymous",
) -> DocumentListResponse:
    conversation, user = _document_scope(conversation_id, user_id)
    try:
        documents = document_ingestion.list_documents(
            user_id=user, conversation_id=conversation
        )
    except DocumentStorageError as exc:
        raise HTTPException(
            status_code=500, detail="Documents could not be listed."
        ) from exc
    return DocumentListResponse(
        documents=[_document_response(metadata, count) for metadata, count in documents]
    )


@app.get("/api/documents/{document_id}", response_model=DocumentManagementResponse)
def get_document(
    document_id: str,
    conversation_id: str,
    user_id: str = "anonymous",
) -> DocumentManagementResponse:
    conversation, user = _document_scope(conversation_id, user_id)
    try:
        metadata, chunk_count = document_ingestion.get_document(
            document_id=document_id,
            user_id=user,
            conversation_id=conversation,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document was not found.") from exc
    except DocumentStorageError as exc:
        raise HTTPException(
            status_code=500, detail="Document could not be loaded."
        ) from exc
    return _document_response(metadata, chunk_count)


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    conversation_id: str,
    user_id: str = "anonymous",
) -> None:
    conversation, user = _document_scope(conversation_id, user_id)
    try:
        document_ingestion.delete_document(
            document_id=document_id,
            user_id=user,
            conversation_id=conversation,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document was not found.") from exc
    except DocumentStorageError as exc:
        raise HTTPException(
            status_code=500, detail="Document could not be deleted."
        ) from exc


def _generate_response(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = conversation_memory.create_conversation()
    elif not conversation_memory.has_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")

    try:
        runtime = runtime_factory.get(
            conversation_memory,
            legacy_manual=conversation_service,
        )
        if hasattr(runtime, "set_user_context"):
            runtime.set_user_context(request.user_id)
        if hasattr(runtime, "respond_with_sources"):
            result = runtime.respond_with_sources(conversation_id, request.message)
            response = result["response"]
            sources = result.get("sources") or None
            return ChatResponse(
                conversation_id=conversation_id,
                response=response,
                sources=sources,
            )
        response = runtime.respond(conversation_id, request.message)
        return ChatResponse(conversation_id=conversation_id, response=response)
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="The Groq service is temporarily rate-limited. Please try again later.",
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="The Groq service authentication is not available.",
        ) from exc
    except BadRequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="The Groq service rejected the request.",
        ) from exc
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The Groq service timed out. Please try again.",
        ) from exc
    except InternalServerError as exc:
        raise HTTPException(
            status_code=502,
            detail="The Groq service is temporarily unavailable.",
        ) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail="The Groq service could not complete the request.",
        ) from exc
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ToolError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the request.",
        ) from exc


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat(request: ChatRequest) -> ChatResponse:
    return _generate_response(request)


@app.post(
    "/api/chat/messages",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    include_in_schema=False,
)
def chat_messages(request: ChatRequest) -> ChatResponse:
    return _generate_response(request)


@app.post("/api/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation() -> dict[str, str]:
    return {"conversation_id": conversation_memory.create_conversation()}


@app.delete("/api/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str) -> None:
    try:
        conversation_memory.clear_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation was not found.") from exc


@app.get("/auth/google")
def google_auth() -> RedirectResponse:
    try:
        url, state = google_oauth.authorization_url()
        oauth_states.add(state)
        return RedirectResponse(url)
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/auth/google/callback")
def google_auth_callback(request: Request) -> dict[str, str]:
    state = request.query_params.get("state")
    if not state or state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid Google OAuth state.")
    try:
        google_oauth.complete(str(request.url), state)
        oauth_states.discard(state)
        return {"status": "authenticated", "message": "Google account connected."}
    except GoogleOAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc