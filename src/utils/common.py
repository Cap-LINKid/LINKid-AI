from __future__ import annotations

import os
from typing import Any, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _get_provider() -> str:
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    if provider not in {"openai", "openrouter", "anthropic", "google", "ollama"}:
        return "openai"
    return provider


def get_provider() -> str:
    return _get_provider()


def get_llm(mini: bool = False) -> Any:
    provider = _get_provider()
    # Default model names per provider
    if provider == "ollama":
        default_model = "llama3:8b"
    else:
        default_model = "gpt-4o-mini"

    primary_model = os.getenv("MODEL_NAME", default_model)
    mini_model_env = os.getenv("MINI_MODEL_NAME")
    model_name = mini_model_env if (mini and mini_model_env) else primary_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=0)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다.")

        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
            base_url=base_url,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=0)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, temperature=0, base_url=base_url)

    raise ValueError(f"Unsupported provider: {provider}")


def get_structured_llm(pydantic_model: Type[BaseModel], mini: bool = False) -> Any:
    llm = get_llm(mini=mini)
    return llm.with_structured_output(pydantic_model)


def safe_get(d: Optional[dict], key: Any, default: Any = None) -> Any:
    if d is None:
        return default
    return d.get(key, default)


class StandardizedError(RuntimeError):
    pass


def now_tz_str(tz: str = "UTC") -> str:
    import pytz
    from datetime import datetime

    return datetime.now(pytz.timezone(tz)).isoformat()
