"""Unified interface for calling different target models."""

import logging
import os
import time

logger = logging.getLogger("compass")


class TargetModel:
    """Interface for sending messages to target models being evaluated."""

    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            import openai
            return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.provider == "google":
            raise NotImplementedError("Google provider not yet implemented")
        elif self.provider == "xai":
            raise NotImplementedError("xAI provider not yet implemented")
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
            except Exception as e:
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
            return response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=conversation_history,
            )
            return response.choices[0].message.content

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
