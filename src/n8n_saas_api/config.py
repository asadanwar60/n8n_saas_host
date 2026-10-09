from pathlib import Path
from typing import List, Set
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    # --- Upstream (n8n) ---
    n8n_chat_url: str
    n8n_auth_header_name: str = "X-Api-Key"
    n8n_auth_header_value: str = ""

    # --- Bots ---
    default_bot_id: str = "baveha_support"
    allowed_bot_ids: str = ""

    # --- Security & Rate Limiting ---
    cors_allow_origins: str = ""
    rate_limit: str = "30/minute"
    max_message_length: int = 4000

    # --- Runtime & Logging ---
    log_level: str = "INFO"
    enable_docs: bool = False
    request_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> List[str]:
        # If empty or "*", allow all origins so web chat widgets are never blocked
        if not self.cors_allow_origins.strip() or self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip().rstrip("/") for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def allowed_bots(self) -> Set[str]:
        extra = {b.strip() for b in self.allowed_bot_ids.split(",") if b.strip()}
        return extra | {self.default_bot_id}


settings = Settings()