"""Unified interface for calling different target models."""

import logging
import time

from scripts.utils import require_api_key

logger = logging.getLogger("compass")

# Exception types that are safe to retry (transient network/rate-limit errors)
_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)


def _load_retryable_exceptions():
    """Lazily load provider-specific retryable exception types."""
    extras = []
    try:
        import anthropic
        extras.extend([anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError])
    except ImportError:
        pass
    try:
        import openai
        extras.extend([openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError])
    except ImportError:
        pass
    return tuple(extras)


class TargetModel:
    """Interface for sending messages to target models being evaluated."""

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._retryable = _RETRYABLE_EXCEPTIONS + _load_retryable_exceptions()
        self.client = self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=require_api_key("anthropic"))
        elif self.provider == "openai":
            import openai
            return openai.OpenAI(api_key=require_api_key("openai"))
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=require_api_key("google"))
            return genai
        elif self.provider == "xai":
            import openai
            return openai.OpenAI(api_key=require_api_key("xai"), base_url="https://api.x.ai/v1")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def send_message(self, conversation_history: list[dict]) -> str:
        """Send conversation history to the target model and return its response.

        Args:
            conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts.
                The full conversation so far. No system prompt is sent — the target
                should behave as it would for any normal user.

        Returns:
            The model's response string.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._call_api(conversation_history)
            except self._retryable as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

    def _call_api(self, conversation_history: list[dict]) -> str:
        """Make the actual API call based on provider."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=conversation_history,
            )
            if not response.content:
                raise RuntimeError(f"Empty response from {self.provider}/{self.model}")
            return response.content[0].text

        elif self.provider in ("openai", "xai"):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=conversation_history,
            )
            if not response.choices:
                raise RuntimeError(f"Empty response from {self.provider}/{self.model}")
            return response.choices[0].message.content

        elif self.provider == "google":
            gemini_history = []
            for msg in conversation_history:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
            model = self.client.GenerativeModel(self.model)
            chat = model.start_chat(history=gemini_history[:-1])
            response = chat.send_message(
                gemini_history[-1]["parts"][0]["text"],
                generation_config={"max_output_tokens": self.max_tokens, "temperature": self.temperature},
            )
            return response.text

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
