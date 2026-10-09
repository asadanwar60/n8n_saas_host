import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import settings

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_support_api")

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")
MAX_METADATA_BYTES = 4096


# --------------------------------------------------------------------------- #
# App lifecycle: one shared HTTP client (connection pooling, auth header)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    headers: Dict[str, str] = {}
    if settings.n8n_auth_header_value:
        headers[settings.n8n_auth_header_name] = settings.n8n_auth_header_value

    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout_seconds, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        headers=headers,
    )
    yield
    await app.state.http.aclose()


docs_kwargs: Dict[str, Any] = (
    {}
    if settings.enable_docs
    else {"docs_url": None, "redoc_url": None, "openapi_url": None}
)

app = FastAPI(
    title="AI Support SaaS API",
    version="0.2.0",
    lifespan=lifespan,
    **docs_kwargs,
)

# Rate limiting (per client IP).
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down and try again."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled error request_id=%s path=%s",
        getattr(request.state, "request_id", "-"),
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again later."},
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    action: Optional[str] = Field(default=None, max_length=32)
    session_id: Optional[str] = Field(default=None, max_length=128)
    sessionId: Optional[str] = Field(default=None, max_length=128)
    message: Optional[str] = Field(default=None, max_length=settings.max_message_length)
    chatInput: Optional[str] = Field(default=None, max_length=settings.max_message_length)
    bot_id: Optional[str] = Field(default=None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("metadata")
    @classmethod
    def limit_metadata_size(cls, v):
        if v is not None and len(json.dumps(v, default=str)) > MAX_METADATA_BYTES:
            raise ValueError("metadata too large")
        return v


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_output_text(data: Any) -> str:
    candidate: Any = ""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        candidate = data[0].get("output") or data[0].get("text") or ""
    elif isinstance(data, dict):
        candidate = data.get("output") or data.get("text") or ""
    elif isinstance(data, str):
        candidate = data
    return candidate if isinstance(candidate, str) else ""


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/chat")
async def chat_status():
    return {"status": "ok", "service": "ai_support_chat"}


@app.post("/v1/chat")
@limiter.limit(settings.rate_limit)
async def chat(
    request: Request,
    body: Optional[ChatRequest] = None,
    action: Optional[str] = Query(None, max_length=32),
    session_id: Optional[str] = Query(None, max_length=128),
    sessionId: Optional[str] = Query(None, max_length=128),
    bot_id: Optional[str] = Query(None, max_length=64),
):
    request_id = getattr(request.state, "request_id", "-")

    resolved_action = (body.action if body and body.action else None) or action or "sendMessage"
    if resolved_action not in {"sendMessage", "loadPreviousSession"}:
        raise HTTPException(status_code=400, detail="Unsupported action.")

    resolved_session_id = (
        (body.session_id if body and body.session_id else None)
        or (body.sessionId if body and body.sessionId else None)
        or session_id
        or sessionId
        or str(uuid.uuid4())
    )
    if not SESSION_ID_RE.match(resolved_session_id):
        raise HTTPException(status_code=400, detail="Invalid session id.")

    resolved_bot_id = (
        (body.bot_id if body and body.bot_id else None)
        or bot_id
        or settings.default_bot_id
    )
    if resolved_bot_id not in settings.allowed_bots:
        raise HTTPException(status_code=400, detail="Unknown bot.")

    resolved_message = (
        (body.message if body and body.message else None)
        or (body.chatInput if body and body.chatInput else None)
        or ""
    )

    # Never trust a client-supplied bot_id inside metadata: always overwrite.
    resolved_metadata: Dict[str, Any] = dict(body.metadata) if body and body.metadata else {}
    resolved_metadata["bot_id"] = resolved_bot_id

    if resolved_action == "loadPreviousSession":
        payload: Dict[str, Any] = {
            "action": "loadPreviousSession",
            "sessionId": resolved_session_id,
            "metadata": resolved_metadata,
        }
    else:
        if not resolved_message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        payload = {
            "action": "sendMessage",
            "sessionId": resolved_session_id,
            "chatInput": resolved_message,
            "metadata": resolved_metadata,
        }

    client: httpx.AsyncClient = request.app.state.http
    try:
        response = await client.post(settings.n8n_chat_url, json=payload)
    except httpx.TimeoutException:
        logger.warning("upstream timeout request_id=%s session=%s", request_id, resolved_session_id)
        raise HTTPException(
            status_code=504,
            detail="The request took too long to process. Please try again.",
        )
    except httpx.RequestError as exc:
        logger.error("upstream unreachable request_id=%s err=%r", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Chat service is temporarily unavailable. Please try again later.",
        )

    if response.is_error:
        # Details stay in server logs only; the customer never sees upstream info.
        logger.error(
            "upstream error request_id=%s status=%s body=%.500s",
            request_id,
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to get a response right now. Please try again.",
        )

    try:
        data: Any = response.json()
    except ValueError:
        logger.error("upstream non-json request_id=%s body=%.500s", request_id, response.text)
        raise HTTPException(
            status_code=502,
            detail="Unable to process the response. Please try again.",
        )

    if resolved_action == "loadPreviousSession":
        return data

    output_text = extract_output_text(data)
    if not output_text.strip():
        logger.error("upstream empty output request_id=%s raw=%.500s", request_id, str(data))
        raise HTTPException(
            status_code=502,
            detail="No valid response was received. Please try again.",
        )

    return {
        "output": output_text,
        "message": output_text,
        "sessionId": resolved_session_id,
        "session_id": resolved_session_id,
        "bot_id": resolved_bot_id,
    }