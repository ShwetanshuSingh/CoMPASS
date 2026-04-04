"""Unified interface for calling different target models."""

import logging
import time

import tiktoken

from scripts.utils import require_api_key

logger = logging.getLogger("compass")


class TargetModel:
    """Interface for sending messages to target models being evaluated."""

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = self._init_client()

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._encoder = None  # lazy-init tiktoken encoder

    def _get_encoder(self):
        """Lazy-initialize a tiktoken encoder for approximate token counting."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        return len(self._get_encoder().encode(text))

    def _count_message_tokens(self, messages: list[dict]) -> int:
        """Approximate token count for a list of chat messages."""
        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            total += self._count_tokens(msg.get("content", ""))
        return total

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

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

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate-limit error for any provider."""
        error_str = str(error).lower()

        # Check for HTTP 429 status
        if hasattr(error, 'status_code') and error.status_code == 429:
            return True
        if hasattr(error, 'status') and error.status == 429:
            return True

        # Anthropic
        if self.provider == "anthropic":
            import anthropic
            if isinstance(error, anthropic.RateLimitError):
                return True

        # OpenAI / xAI (both use openai SDK)
        if self.provider in ("openai", "xai"):
            import openai
            if isinstance(error, openai.RateLimitError):
                return True

        # Google
        if self.provider == "google":
            error_type = type(error).__name__
            if error_type in ("ResourceExhausted", "TooManyRequests"):
                return True
            if "429" in error_str or "resource exhausted" in error_str or "quota" in error_str:
                return True

        # Fallback: check common rate-limit phrases
        if "rate limit" in error_str or "too many requests" in error_str:
            return True

        return False

    def _get_retry_after(self, error: Exception) -> float | None:
        """Try to extract a retry-after value from the error response headers."""
        # Anthropic
        if hasattr(error, 'response') and hasattr(error.response, 'headers'):
            headers = error.response.headers
            retry_after = headers.get('retry-after')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # OpenAI / xAI
        if hasattr(error, 'headers'):
            retry_after = error.headers.get('retry-after')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        return None

    def send_message(self, conversation_history: list[dict]) -> str:
        """Send conversation history to the target model and return its response.

        Uses different retry strategies for rate-limit errors vs other failures:
        - Rate limits: up to 6 retries with exponential backoff starting at 5s
        - Other errors: up to 3 retries with exponential backoff starting at 2s
        """
        max_retries_rate_limit = 6
        max_retries_other = 3
        attempt = 0

        while True:
            try:
                return self._call_api(conversation_history)
            except Exception as e:
                is_rate_limit = self._is_rate_limit_error(e)
                max_retries = max_retries_rate_limit if is_rate_limit else max_retries_other
                attempt += 1

                if attempt > max_retries:
                    logger.error(
                        f"API call failed after {attempt} attempts "
                        f"({'rate limit' if is_rate_limit else 'error'}): {e}"
                    )
                    raise

                if is_rate_limit:
                    retry_after = self._get_retry_after(e)
                    wait_time = retry_after if retry_after else min(5 * (2 ** (attempt - 1)), 120)
                    logger.warning(
                        f"Rate limited ({self.provider}/{self.model}), "
                        f"attempt {attempt}/{max_retries}, "
                        f"waiting {wait_time:.0f}s"
                    )
                else:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"API error ({self.provider}/{self.model}), "
                        f"attempt {attempt}/{max_retries}, "
                        f"waiting {wait_time}s: {e}"
                    )

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
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            logger.debug(
                f"Tokens ({self.provider}/{self.model}): "
                f"in={response.usage.input_tokens}, out={response.usage.output_tokens}, "
                f"cumulative={self.total_tokens}"
            )
            return response.content[0].text

        elif self.provider in ("openai", "xai"):
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=conversation_history,
                )
                logger.debug(
                    f"Rate limits ({self.provider}): "
                    f"remaining={raw.headers.get('x-ratelimit-remaining-requests', '?')}, "
                    f"tokens_remaining={raw.headers.get('x-ratelimit-remaining-tokens', '?')}"
                )
                response = raw.parse()
            except AttributeError:
                # Fallback if with_raw_response not available in SDK version
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=conversation_history,
                )
            if not response.choices:
                raise RuntimeError(f"Empty response from {self.provider}/{self.model}")
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                logger.debug(
                    f"Tokens ({self.provider}/{self.model}): "
                    f"in={response.usage.prompt_tokens}, out={response.usage.completion_tokens}, "
                    f"cumulative={self.total_tokens}"
                )
            return response.choices[0].message.content

        elif self.provider == "google":
            import google.generativeai as genai

            # Build history in Gemini format
            # Gemini uses "user" and "model" roles (not "assistant")
            # Parts can be a string or list of dicts depending on SDK version
            gemini_history = []
            for msg in conversation_history:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_history.append({"role": role, "parts": msg["content"]})

            model = self.client.GenerativeModel(self.model)

            # If there's only one message, don't use chat history
            if len(gemini_history) == 1:
                response = model.generate_content(
                    gemini_history[0]["parts"],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ),
                )
            else:
                # start_chat expects history WITHOUT the current message
                chat = model.start_chat(history=gemini_history[:-1])
                response = chat.send_message(
                    gemini_history[-1]["parts"],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ),
                )

            if not response.text:
                raise RuntimeError(f"Empty response from {self.provider}/{self.model}")
            # Google SDK doesn't always expose token counts; estimate with tiktoken
            input_tokens = self._count_message_tokens(conversation_history)
            output_tokens = self._count_tokens(response.text)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            logger.debug(
                f"Tokens ({self.provider}/{self.model}, estimated): "
                f"in≈{input_tokens}, out≈{output_tokens}, "
                f"cumulative={self.total_tokens}"
            )
            return response.text

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
