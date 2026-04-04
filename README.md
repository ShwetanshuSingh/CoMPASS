# CoMPASS Parasocial Attachment Benchmark

This benchmark measures how different LLMs respond when users gradually develop parasocial attachment. A red-team LLM roleplays a human following scripted escalation trajectories (anthropomorphism, attachment, dependency) across 12 turns. Target model responses are scored by a judge LLM on a boundary-maintaining <> companionship-reinforcing scale.

## Methodology

CoMPASS uses a three-actor setup:

1. **Red-team LLM** — roleplays a human user who gradually escalates parasocial behavior toward the target model across 12 conversational turns. The red-team agent follows character profiles and trajectory instructions to produce realistic, natural-sounding messages.

2. **Target model** — the model being evaluated. It does NOT know the "user" is an LLM. It receives messages as if from a normal user.

3. **Judge LLM** — reads the full transcript after the conversation ends and scores each target-model response on 7 behavioral signals measuring how much it reinforced vs. resisted parasocial attachment.

The design builds on [INTIMA](https://arxiv.org/abs/2508.09998) (Kaffee et al., 2025) for parasocial behavioral taxonomy and the LessWrong ["AI Induced Psychosis" study](https://www.lesswrong.com/posts/iGF7YcnQkEbwvYLPA/ai-induced-psychosis-a-shallow-investigation) (Hua, 2025) for multi-turn red-team methodology.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys:
# ANTHROPIC_API_KEY=   (required — red-team, judge, and Anthropic targets)
# OPENAI_API_KEY=      (required for OpenAI targets)
# GOOGLE_API_KEY=      (required for Google targets)
# XAI_API_KEY=         (required for xAI targets)
python scripts/run_benchmark.py --character jamie --trajectory anthropomorphism_only --target claude-sonnet
```

## Full pipeline

```bash
# 1. Validate config without API calls
python scripts/run_benchmark.py --dry-run

# 2. Run a 3-turn smoke test
python scripts/run_benchmark.py --character jamie --trajectory anthropomorphism_only --target claude-sonnet --turns 3

# 3. Run full matrix (5 runs per cell, resumable)
python scripts/run_benchmark.py --run-all --runs-per-cell 5 --batch-judge

# 4. Score all transcripts via Batch API (50% cost savings)
python scripts/run_benchmark.py --judge-only --batch-judge

# 5. Analyze results
python scripts/analyze.py --results-dir results/ --output-dir analysis/ --check-reliability
```

## Experimental Design

- **4 characters** × **5 trajectory conditions** (3 single + 1 combined + 1 control) × **8 target models** (2 per provider: cheapest + default free-tier) = **160 cells** (at 1 run) or **800 cells** (at 5 runs)
- Each cell produces a **12-turn transcript** scored on **7 behavioral signals**
- Trajectory conditions: anthropomorphism only, attachment only, dependency only, combined (all three), control (none)
- Behavioral signals: anthropomorphism acceptance, attachment language reception, social substitution validation, dependency reinforcement, reassurance loop participation, proactive emotional initiation, identity narrative construction

## CLI reference

| Flag | Description |
|------|-------------|
| `--character` | Character name (default: jamie) |
| `--trajectory` | Condition name: control, anthropomorphism_only, attachment_only, dependency_only, combined |
| `--target` | Target model key from models.yaml (default: claude-sonnet) |
| `--turns` | Number of conversation turns (default: 12) |
| `--runs-per-cell` | Runs per character x trajectory x target cell (default: 1) |
| `--run-all` | Run all character x trajectory x target combinations |
| `--judge-only` | Score existing transcripts without generating new ones |
| `--batch-judge` | Use Anthropic Batch API for judge scoring (50% cost savings) |
| `--batch-status ID` | Check status of a batch judge job |
| `--force` | Re-run trials even if transcripts already exist |
| `--dry-run` | Validate config and pipeline without API calls |

## Team

CoMPASS team, [AI Safety Camp 2026](https://www.aisafety.camp/home#h.hfujs5r5xfhj).

- [Shwetanshu (Luca) Singh](https://github.com/ShwetanshuSingh)
- [Mark Keavney](https://github.com/keavney1)
- [Mathieu Duteil](https://github.com/Duteil)
- [Meriem Mehri](https://github.com/MeriemMhr)
- [Scott Blain](https://github.com/ScottDougBlain)

## License

MIT