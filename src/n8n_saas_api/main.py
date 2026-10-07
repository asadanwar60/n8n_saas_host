from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings


app = FastAPI(
    title="AI Support SaaS API",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    bot_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10000)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ai_support_api",
    }


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    n8n_url = settings.n8n_chat_url

    payload = {
        "sessionId": request.session_id,
        "action": "sendMessage",
        "chatInput": request.message,
        "metadata": {
            "bot_id": request.bot_id,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                n8n_url,
                json=payload,
            )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The request took too long to process. Please try again.",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Chat service is temporarily unavailable. Please try again later.",
        )

    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail="Unable to get a response for your message.",
        )

    try:
        data: Any = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Unable to process the response. Please try again.",
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail="Unable to process the response. Please try again.",
        )

    message = data.get("output")

    if not isinstance(message, str) or not message.strip():
        raise HTTPException(
            status_code=502,
            detail="No valid response was received. Please try again.",
        )

    return {
        "session_id": request.session_id,
        "message": message,
    }