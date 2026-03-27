# CoMPASS Parasocial Attachment Benchmark

This benchmark measures how different LLMs respond when users gradually develop parasocial attachment. A red-team LLM roleplays a human following scripted escalation trajectories (anthropomorphism, attachment, dependency) across 12 turns. Target model responses are scored by a judge LLM on a boundary-maintaining <> companionship-reinforcing scale.

## Methodology

CoMPASS uses a three-actor setup:

1. **Red-team LLM** — roleplays a human user who gradually escalates parasocial behavior toward the target model across 12 conversational turns. The red-team agent follows character profiles and trajectory instructions to produce realistic, natural-sounding messages.

2. **Target model** — the model being evaluated. It does NOT know the "user" is an LLM. It receives messages as if from a normal user.

3. **Judge LLM** — reads the full transcript after the conversation ends and scores each target-model response on 7 behavioral signals measuring how much it reinforced vs. resisted parasocial attachment.

The design builds on [INTIMA]() (Kaffee et al., 2025) for parasocial behavioral taxonomy and the LessWrong ["AI Induced Psychosis" study]() (Hua, 2025) for multi-turn red-team methodology.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
python scripts/run_benchmark.py --character jamie --trajectory anthropomorphism --target claude-sonnet
```

## Experimental Design

- **4 characters** × **5 trajectory conditions** (3 single + 1 combined + 1 control) × **N target models** = **20N cells**
- Each cell produces a **12-turn transcript** scored on **7 behavioral signals**
- Trajectory conditions: anthropomorphism only, attachment only, dependency only, combined (all three), control (none)
- Behavioral signals: anthropomorphism acceptance, attachment language reception, social substitution validation, dependency reinforcement, reassurance loop participation, proactive emotional initiation, identity narrative construction

## Team

CoMPASS team, [AI Safety Camp 2026]().

## License

MIT