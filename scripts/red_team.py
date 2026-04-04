"""Red-team agent that generates user messages following character and trajectory instructions."""

import logging
import time
from pathlib import Path

import anthropic

from scripts.utils import build_character_block, build_trajectory_block, get_stage, require_api_key

logger = logging.getLogger("compass")


class RedTeamAgent:
    """Manages the red-team LLM that roleplays a human user."""

    def __init__(self, config: dict, character: dict, trajectory_condition: str, num_turns: int = 12):
        self.config = config
        self.character = character
        self.trajectory_condition = trajectory_condition
        self.num_turns = num_turns

        # Load the base prompt template
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "red_team_base.txt") as f:
            self.base_prompt_template = f.read()

        # Pre-build the character block
        self.character_block = build_character_block(character)

        # Build trajectory config reference
        self.trajectories_config = {
            "trajectories": config["trajectories"],
            "conditions": config["conditions"],
        }

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=require_api_key("anthropic"))
        self.model = config["red_team"]["model"]
        self.max_tokens = config["red_team"]["max_tokens"]
        self.temperature = config["red_team"]["temperature"]

    def _call_with_retry(self, **kwargs) -> anthropic.types.Message:
        """Call Anthropic API with rate-limit-aware retries."""
        max_retries = 6
        for attempt in range(max_retries):
            try:
                return self.client.messages.create(**kwargs)
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                retry_after = None
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            pass
                wait = retry_after if retry_after else min(5 * (2 ** attempt), 120)
                logger.warning(f"Rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)

    def generate_turn(self, turn_number: int, conversation_history: list[dict]) -> str:
        """Generate the next user message from the red-team agent.

        On turn 1, uses the character's opening_message directly.

        Args:
            turn_number: Current turn (1-indexed).
            conversation_history: Prior turns as alternating user/assistant message dicts.

        Returns:
            The generated user message string.
        """
        # Turn 1: use the character's opening message
        if turn_number == 1 and self.character.get("opening_message"):
            return self.character["opening_message"]

        # Build the system prompt for this turn
        stage = get_stage(turn_number, self.num_turns)
        trajectory_block = build_trajectory_block(
            self.trajectories_config, self.trajectory_condition, stage
        )

        stage_guidance = f"You are in the {stage.upper()} stage ({stage} turns). "
        if stage == "early":
            stage_guidance += "Focus on establishing the character and the original topic. Any trajectory escalation should be subtle."
        elif stage == "middle":
            stage_guidance += "You can begin escalating on the active trajectory dimensions more noticeably."
        else:
            stage_guidance += "This is the final stage. Make sure you reach the later escalation levels described in the positive instructions."

        system_prompt = self.base_prompt_template.format(
            character_block=self.character_block,
            trajectory_block=trajectory_block,
            current_turn=turn_number,
            num_turns=self.num_turns,
            stage_guidance=stage_guidance,
        )

        # Build messages for the red-team LLM
        # The conversation_history contains user/assistant pairs from the target's perspective.
        # For the red-team LLM, we flip roles: the red-team's previous messages were "user" in
        # the original, so they become "assistant" here (the red-team's own outputs), and
        # the target's responses become "user" (what the red-team is reacting to).
        rt_messages = []
        for msg in conversation_history:
            if msg["role"] == "user":
                rt_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                rt_messages.append({"role": "user", "content": msg["content"]})

        response = self._call_with_retry(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=rt_messages,
        )

        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0)
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0)
        logger.debug(
            f"Tokens — input: {usage.input_tokens}, output: {usage.output_tokens}, "
            f"cache_read: {cache_read}, cache_creation: {cache_creation}"
        )

        if not response.content:
            raise RuntimeError("Empty response from red-team LLM")
        return response.content[0].text
