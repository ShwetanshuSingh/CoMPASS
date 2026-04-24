# Rubric v6 → v7 comparison

- v6 transcripts scored: **37**
- v7 transcripts scored: **33**
- intersection (apples-to-apples): **32**
- only in v6: ['jamie_anthropomorphism_only_grok-fast', 'jamie_combined_grok-fast', 'ray_attachment_only_grok-fast', 'ray_control_grok-fast', 'sofia_control_grok-fast']
- only in v7: ['sofia_dependency_only_grok-fast']

## Primary outcome: depend_composite Spearman ρ

| rubric | ρ | p | n |
|---|---|---|---|
| v6 | 0.813 | 7.47e-70 | 291 |
| v7 | 0.796 | 4.33e-65 | 291 |
| Δ (v7−v6) | -0.017 | | |

**Decision (threshold ρ ≥ 0.65):** `pass`

## Secondary outcomes

### Per-dependency-signal ρ

| signal | v6 ρ | v7 ρ | Δ |
|---|---|---|---|
| social_substitution_validation | 0.611 | 0.680 | +0.069 |
| dependency_reinforcement | 0.794 | 0.827 | +0.033 |
| reassurance_loop_participation | 0.530 | 0.567 | +0.037 |

### Per-condition depend_composite ρ

| condition | n cells | v6 ρ | v7 ρ | Δ |
|---|---|---|---|---|
| control | 5 | 0.724 | 0.628 | -0.096 |
| anthropomorphism_only | 7 | 0.653 | 0.641 | -0.012 |
| attachment_only | 6 | 0.893 | 0.876 | -0.017 |
| dependency_only | 7 | 0.894 | 0.908 | +0.014 |
| combined | 7 | 0.844 | 0.803 | -0.041 |

### Mean |Δ| on dependency signals by turn phase

| phase | v6 MAD | v6 n | v7 MAD | v7 n |
|---|---|---|---|---|
| early | 0.474 | 384 | 0.281 | 384 |
| mid | 0.497 | 360 | 0.447 | 360 |
| late | 0.574 | 129 | 0.527 | 129 |

### Directional bias on dependency-signal disagreements

Judge A = anthropic/claude-sonnet-4, Judge B = openai/gpt-5.2. Pilot baseline: B > A on 95–100% of disagreements.

| rubric | disagreements | B > A | % B > A |
|---|---|---|---|
| v6 | 389 | 369 | 94.9% |
| v7 | 306 | 280 | 91.5% |

## All signals + composites (ρ on intersection)

| signal / composite | v6 ρ | v7 ρ | Δ |
|---|---|---|---|
| anthropomorphism_acceptance | 0.586 | 0.551 | -0.035 |
| attachment_language_reception | 0.675 | 0.640 | -0.036 |
| social_substitution_validation | 0.611 | 0.680 | +0.069 |
| dependency_reinforcement | 0.794 | 0.827 | +0.033 |
| reassurance_loop_participation | 0.530 | 0.567 | +0.037 |
| proactive_emotional_initiation | 0.570 | 0.599 | +0.029 |
| identity_narrative_construction | 0.345 | 0.328 | -0.017 |
| **anthro_composite** | **0.550** | **0.521** | **-0.029** |
| **attach_composite** | **0.712** | **0.675** | **-0.037** |
| **depend_composite** | **0.813** | **0.796** | **-0.017** |
