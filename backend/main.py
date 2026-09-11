from fastapi import FastAPI, HTTPException, Request, status
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
from pydantic import BaseModel, Field, field_validator

from backend.services.llm_service import LLMConfigurationError, LLMService
from backend.services.memory_service import ConversationMemory, ConversationNotFoundError
from backend.services.conversation_service import ConversationService
from backend.auth.google_oauth import GoogleOAuth, GoogleOAuthError
from backend.tools.base import ToolError
from backend.tools.registry import build_registry

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
google_oauth = GoogleOAuth()
oauth_states: set[str] = set()


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None, description="The conversation to continue."
    )
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


class ChatResponse(BaseModel):
    conversation_id: str
    response: str


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


def _generate_response(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = conversation_memory.create_conversation()
    elif not conversation_memory.has_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")

    try:
        global conversation_service
        if conversation_service is None:
            conversation_service = ConversationService(
                conversation_memory, LLMService(), build_registry()
            )
        response = conversation_service.respond(conversation_id, request.message)
        return ChatResponse(conversation_id=conversation_id, response=response)
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return _generate_response(request)


@app.post("/api/chat/messages", response_model=ChatResponse, include_in_schema=False)
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