"""OpenRouter model (OpenAI compatible API)"""
import os
import time
from typing import Optional
from .openai_base import OpenAIBaseModel


class OpenRouterChatbot(OpenAIBaseModel):
    """OpenRouter chatbot model"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenRouter model

        Args:
            model_name: Model name, defaults to LLM_MODEL env var or "google/gemma-3-4b-it"
            device: Device (kept for interface consistency)
            api_key: API key, if None get from OPENROUTER_API_KEY / OPENAI_API_KEY
            base_url: API base URL, defaults to https://openrouter.ai/api/v1
        """
        # Determine model name
        resolved_model = (
            model_name
            if (model_name and model_name.lower() != "openrouter")
            else os.getenv("LLM_MODEL", "google/gemma-3-4b-it")
        )

        resolved_api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("\ufeffOPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        resolved_base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )

        super().__init__(
            model_name=resolved_model,
            device=device,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            env_api_key_name="OPENROUTER_API_KEY",
            default_base_url="https://openrouter.ai/api/v1",
        )

    def generate_response(
        self,
        user_input: str,
        max_length: int = 4096,
        temperature: float = 0.1,
        retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> str:
        """
        Generate response with retry logic for OpenRouter API.
        """
        last_exception = None
        effort = os.getenv("OPENROUTER_REASONING_EFFORT", "none")
        extra_body = {"reasoning": {"effort": effort}} if effort else None

        for attempt in range(retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": user_input}],
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "stream": False,
                    "extra_headers": {
                        "HTTP-Referer": "https://github.com/DEEP-PolyU/LegalGraphRAG",
                        "X-Title": "LegalGraphRAG-AutoTOR",
                    },
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as api_err:
                    # If model doesn't accept reasoning param, retry without it
                    if extra_body and "reasoning" in str(api_err).lower():
                        extra_body = None
                        kwargs.pop("extra_body", None)
                        response = self.client.chat.completions.create(**kwargs)
                    else:
                        raise api_err

                msg = response.choices[0].message
                content = msg.content or ""
                if not content and hasattr(msg, "reasoning") and msg.reasoning:
                    content = msg.reasoning
                return content
            except Exception as e:
                last_exception = e
                if attempt < retries - 1:
                    time.sleep(backoff_seconds * (attempt + 1))
                else:
                    print(f"OpenRouter API call failed after {retries} attempts: {e}")

        return ""
