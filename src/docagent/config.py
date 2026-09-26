"""Settings read from environment variables or a local ``.env`` file.

API keys live in ``.env`` (git-ignored); ``.env.example`` documents every
variable with a placeholder value.

The LLM is any provider exposing the OpenAI-compatible chat completions API
(Mistral, Groq, OpenRouter, a local Ollama, vLLM...). ``LLM_PROVIDER`` picks a
preset (URL, default model, which key to use); ``LLM_BASE_URL``, ``LLM_MODEL``
and ``LLM_API_KEY`` override it.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["mistral", "groq", "openrouter", "ollama", "openai_compatible"]


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    model: str | None  # None: LLM_MODEL must be set (model lists change often)
    needs_key: bool = True


PRESETS: dict[str, ProviderPreset] = {
    "mistral": ProviderPreset("https://api.mistral.ai/v1", "mistral-small-latest"),
    "groq": ProviderPreset("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openrouter": ProviderPreset("https://openrouter.ai/api/v1", None),
    "ollama": ProviderPreset("http://localhost:11434/v1", "qwen2.5:7b", needs_key=False),
    "openai_compatible": ProviderPreset(None, None),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "mistral"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    # Provider-specific keys, so several can sit in .env and LLM_PROVIDER switches.
    mistral_api_key: SecretStr | None = None
    mistral_model: str | None = None  # kept for compatibility with earlier .env files
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    llm_timeout_s: float = 120.0
    llm_max_retries: int = 8
    # Minimum pause between two requests: free tiers allow ~1 request/second.
    llm_min_interval_s: float = 1.5
    # Characters of source text per request; a longer page is split.
    llm_batch_chars: int = 6000
    # Blocks per request: pages of chart labels have many tiny blocks, and small
    # models tend to stop before the last ids of a long list.
    llm_batch_items: int = 40

    # Imposed terminology: GLOSSARY_PATH for one file, otherwise
    # <glossary_dir>/glossary_<source>_<target>.json when it exists.
    glossary_path: Path | None = None
    glossary_dir: Path | None = Path("data")
    translation_cache: Path | None = Path(".cache/translations.json")

    # "llm" for real translations, "fake" to demo the layout without any API key.
    translator: Literal["llm", "fake"] = "llm"

    # API server
    max_upload_mb: int = 20
    max_pages: int = 50
    jobs_dir: Path = Path(".jobs")

    # -- resolved LLM settings ---------------------------------------------------

    @property
    def preset(self) -> ProviderPreset:
        return PRESETS[self.llm_provider]

    @property
    def resolved_base_url(self) -> str:
        url = self.llm_base_url or self.preset.base_url
        if not url:
            raise RuntimeError("LLM_BASE_URL is required for this provider (see .env.example)")
        return url.rstrip("/")

    @property
    def resolved_model(self) -> str:
        model = self.llm_model
        if not model and self.llm_provider == "mistral":
            model = self.mistral_model
        model = model or self.preset.model
        if not model:
            raise RuntimeError("LLM_MODEL is required for this provider (see .env.example)")
        return model

    @property
    def resolved_api_key(self) -> SecretStr | None:
        specific = {
            "mistral": self.mistral_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(self.llm_provider)
        key = self.llm_api_key or specific
        if key is None and self.preset.needs_key:
            name = f"{self.llm_provider.upper()}_API_KEY or LLM_API_KEY"
            raise RuntimeError(f"{name} is not set (put it in .env, see .env.example)")
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()
