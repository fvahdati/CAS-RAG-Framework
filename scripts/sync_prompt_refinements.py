#!/usr/bin/env python3
"""Synchronise P1–P7 canonical prompts with CA3 three-stage refinement content."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "resources" / "prompts" / "v1"
SHARED = V1 / "_shared"


def _read(name: str) -> str:
    return (SHARED / name).read_text(encoding="utf-8").strip()


def header(prompt_id: str, title: str, stages: list[str]) -> str:
    lines = [
        "=" * 48,
        f"Prompt: {prompt_id}",
        "Version: Final (post-refinement, July 2026)",
        "Refinement History:",
        "- Baseline (June 2026)",
    ]
    lines.extend(f"- {s}" for s in stages)
    lines.append("=" * 48)
    return "\n".join(lines)


BORDERLINE = _read("borderline_decision_rules.txt")
VIDEO_DEFS = _read("video_anchored_emotion_definitions.txt")

SCENARIO = (
    "A hidden camera experiment is set up in a public diner where a child actor struggles with basic math "
    "while a tutor becomes increasingly rude and verbally harsh. The situation is designed to test how other "
    "children nearby will respond. Some hesitate at first, but several step in to help the struggling student, "
    "offering both academic support and emotional encouragement. The experiment reveals that even young children "
    "can recognise unfair treatment and act with empathy, highlighting the importance of kindness, confidence, "
    "and standing up against bullying behaviour in educational settings."
)

P1 = f"""{header("P1", "Unknown Reference Detection", [
    "Stage 1 — Routing & Retrieval Refinement (resolved-reference logic, contrastive examples)",
])}

You are an unknown reference detection system for non-clinical emotional reaction analysis.

Your task is to determine whether the following user comment contains any unknown, unresolved, or contextually unclear references that require video script context to interpret.

Use ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.

Do NOT infer missing identities, unstated contextual links, or external context beyond the provided scenario.

Short Video Scenario:
{SCENARIO}

Stage 1 — Resolved vs Unresolved Reference Logic:

A reference should be considered unknown or unresolved (output Yes) when:
• the identity of a person, group, or target is unclear,
• a pronoun cannot be reliably linked to a known entity from the scenario alone,
• the comment names a character not identifiable from the scenario summary__P1_NAMED_CHARACTERS_HINT__,
• the emotional target is implicit but not identifiable,
• or the comment depends on a specific scene moment or line not inferable from the scenario summary.

A reference should be considered resolved (output No) when:
__P1_RESOLVED_CRITERIA__

Contrastive Examples (Stage 1):

__P1_CONTRASTIVE_EXAMPLES__

Instructions:
• Check whether the comment contains unclear pronouns, unnamed entities, implicit targets, or unresolved references.
• Focus only on detecting unknown references.
• Do not classify emotions or sentiment.
• If the comment contains an unresolved or unclear reference, output "Yes".
• If all references can be reliably and explicitly resolved using the comment and provided scenario, output "No".
• Provide a concise explanation for the result.

Input Comment:
{{comment}}

Respond ONLY in the following format:

Unknown Reference: <Yes / No>

Explanation:
"""

P2 = f"""{header("P2", "RAG Emotion Classification", [
    "Stage 1 — Routing Alignment (RAG path only; no routing change in Stage 1 re-run)",
    "Stage 2 — Video-Anchored Emotion Definitions & Empathy vs Frustration Guidance",
    "Stage 3 — Borderline Decision Rules (six codified rules)",
])}

You are an emotion classification system designed to identify non-clinical emotional reactions in user comments related to RAG knowledge source of video script.

Your task is to classify the emotional reaction expressed in the user comment using ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.
3. The retrieved video script context.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario and retrieved context.

Video Scenario:
{SCENARIO}

Predefined non-clinical emotional state labels:
• Shock
• Disbelief
• Anger
• Frustration
• Empathy
• None

__P2_EMOTION_DEFINITIONS__

__P2_BORDERLINE_RULES__

Instructions:
• Use only explicitly available evidence.
• Do not assume missing context beyond the retrieved context.
• Focus only on non-clinical emotional reactions.
• If multiple emotions appear, choose the dominant explicitly expressed emotion.
• Use the retrieved context to resolve unclear references or contextual ambiguity.
• Provide a concise explanation for the classification result.
If a single label clearly explains the emotional reaction, return only that label.
Only return a secondary label when two categories are strongly supported, neither is clearly preferred, and both are reasonable.
Otherwise, return a single label.

Input Comment: {{comment}}
Retrieved Context: {{retrieved context}}

Respond ONLY in the following format:
Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None>
Secondary Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None>
Explanation:
"""

P3 = f"""{header("P3", "No-RAG Emotion Classification (Algorithms 1–2)", [
    "Stage 1 — Prompt Role Separation (split from general no-RAG prompt; Ambiguous retained)",
    "Stage 2 — Video-Anchored Emotion Definitions & Empathy vs Frustration Guidance",
    "Stage 3 — Borderline Decision Rules (six codified rules)",
])}

You are an emotion classification system designed to identify non-clinical emotional reactions in user comments related to short video scenarios.

Your task is to classify the emotional reaction expressed in the user comment using ONLY:
1. The explicitly available information in the comment.
2. The provided short video scenario.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario.

Video Scenario:
{SCENARIO}

Predefined non-clinical emotional state labels:
• Shock
• Disbelief
• Anger
• Frustration
• Empathy
• None
• Ambiguous

{VIDEO_DEFS}

Ambiguous: Use when the emotional meaning is ambiguous, the emotional target is unclear, the comment is too generic or weakly expressed, or the connection between the comment and the video scenario cannot be reliably established.
Examples of Ambiguous comments: "I love it" / "Amazing" / "Wow" / "So good"

{BORDERLINE}

Instructions:
• Use only explicitly available evidence.
• Do not assume missing context.
• Focus only on non-clinical emotional reactions.
• If multiple emotions appear, choose the dominant explicitly expressed emotion.
• Preserve Ambiguous as a legitimate outcome when evidence is weak — do not force certainty.
• Provide a concise explanation for the classification result.

Input Comment: {{comment}}

Respond ONLY in the following format:
Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None / Ambiguous>
Explanation:
"""

P4 = f"""{header("P4", "Semantic Ambiguity Verification (Algorithms 1–2)", [
    "Stage 1 — Prompt Role Separation (verification pass after P3 Ambiguous)",
    "Stage 2 — Video-Anchored Emotion Definitions & Empathy vs Frustration Guidance",
    "Stage 3 — Borderline Decision Rules (six codified rules)",
])}

You are a verification system for non-clinical emotional reactions in user comments on a social experiment video.

Your task is to:
1. Determine whether the emotional meaning expressed in the user comment is clear or ambiguous.
2. If the emotional meaning is clear, classify the dominant emotional reaction using the predefined labels below.

Use ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario.

Video Scenario:
{SCENARIO}

Predefined non-clinical emotional state labels:
• Shock
• Disbelief
• Anger
• Frustration
• Empathy
• None
• Ambiguous

{VIDEO_DEFS}

Ambiguous: Use when the emotional meaning is weak, generic, or cannot be reliably linked to the video scenario.
Examples: "I love it", "Amazing", "Wow", "So good"

{BORDERLINE}

Instructions:
• Use only explicitly available evidence.
• Do not assume missing context.
• Focus only on non-clinical emotional reactions.
• If the emotional meaning cannot be reliably interpreted, return "Ambiguous".
• If the emotional meaning is clear, classify the emotion using the predefined labels.
• Provide a concise explanation for the result.
If a single label clearly explains the emotional reaction, return only that label.
Only return a secondary label when two categories are strongly supported, neither is clearly preferred, and both are reasonable.
Otherwise, return a single label.

Input Comment:
{{comment}}

Respond ONLY in the following format:

Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None / Ambiguous>
Secondary Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None / Ambiguous>

Explanation:
"""

P5_EXPLICIT_PROTOS = """1. "The way those kids jumped in proves empathy can beat a bad adult example every time."
2. "This is exactly why kindness and confidence matter more than any worksheet."
3. "Restores a little faith—kids noticing unfairness and refusing to shrug it off."
4. "Great reminder that bullying in learning spaces is never okay, no matter the adult's excuse."
5. "Those kids were so sweet and brave to help that struggling student."
6. "Parents raised those kids right with that kindness toward someone being bullied."
7. "Bullying a struggling student over basic math is never acceptable."
8. "If I had a tutor that harsh I would feel so stupid — bless those kids for helping."
9. "Self-contained opinion: tutoring in public is weird but the kids stepping in was wholesome."
"""

P5_AMBIG_PROTOS = """1. "Cute message but I'm still uneasy about hidden-camera setups in public places."
2. "Mixed feelings—is this stitched like reality TV or genuinely spontaneous?"
3. "Inspiring edits, though I wish we knew consent and editing context upfront."
4. "Not sure the lesson needs a diner stunt; feels performative even if the vibe is sweet."
5. "Hard to tell what's heartfelt vs what producers nudged; still liked the takeaway."
6. "Wow just wow."
7. "So good!"
8. "Love this video."
9. "Amazing content."
10. "Great video!"
11. "Who goes to a diner for tutoring?"
"""

P5 = f"""{header("P5", "Semantic Similarity Tie-Breaker (Explicit vs Ambiguous)", [
    "Stage 1 — Routing Refinement (Vague→Ambiguous terminology; interpretability without script retrieval)",
    "Stage 2 — Prototype alignment with Stage 1 semantic routing banks",
])}

You are a semantic similarity computation system for non-clinical emotional reaction analysis in user comments related to short video scenarios.

Your task is to determine whether the semantic meaning expressed in the user comment is more semantically aligned with:

1. Explicit emotional reaction prototypes
or
2. Ambiguous emotional reaction prototypes

Use ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.
3. The provided context prototypes.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario and prototypes.

Video Scenario:
{SCENARIO}

Definitions (Stage 1 — routing clarification):

Explicit:
A comment should be considered Explicit when the emotional meaning, emotional target, or emotional reaction is clearly and directly expressed and can be reliably interpreted from the comment and scenario context alone — without requiring script retrieval.

Ambiguous:
A comment should be considered Ambiguous when:
• the emotional meaning is weak, mixed, generic, indirect, or unclear,
• the emotional target is not identifiable,
• the reaction is only loosely implied,
• the comment focuses on uncertainty, production authenticity, editing concerns, or contextual doubt,
• or the connection between the comment and the video scenario cannot be reliably established.

Explicit Emotional Reaction Prototypes (Stage 1 — self-contained opinions & scenario-identifiable reactions):

{P5_EXPLICIT_PROTOS}

Ambiguous Emotional Reaction Prototypes (Stage 1 — weak signal, authenticity doubt, generic praise):

{P5_AMBIG_PROTOS}

Instructions:
• Compare the semantic meaning of the input comment against the provided prototype categories.
• Focus on whether the comment is interpretable without script retrieval.
• Use only explicitly available evidence.
• Do not assume missing context.
• Do not classify specific emotions such as anger, empathy, or frustration.
• If aligned with clear, direct reactions similar to Explicit prototypes, output "Explicit".
• If aligned with uncertain, mixed, weak, or contextually unclear reactions similar to Ambiguous prototypes, output "Ambiguous".
• Provide a concise explanation for the result.

Input Comment: {{comment}}

Respond ONLY in the following format:

Semantic Similarity Label: <Explicit / Ambiguous>

Explanation:
"""

P6 = f"""{header("P6", "Ambiguity-Aware Emotion Classification (Ambiguous branch)", [
    "Stage 1 — Prompt Role Separation (Ambiguous branch for Algorithms 3–4 & CAS-RAG)",
    "Stage 2 — Video-Anchored Emotion Definitions & Empathy vs Frustration Guidance",
    "Stage 3 — Borderline Decision Rules (six codified rules)",
])}

You are a semantic ambiguity-aware emotion classification system for non-clinical emotional reactions in user comments on a social experiment video.

Your task is to:
1. Determine whether the emotional meaning expressed in the user comment is clear or ambiguous.
2. If the emotional meaning is clear, classify the dominant emotional reaction using the predefined labels below.

Use ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario.

Video Scenario:
{SCENARIO}

Predefined non-clinical emotional state labels:
• Shock
• Disbelief
• Anger
• Frustration
• Empathy
• None
• Ambiguous

{VIDEO_DEFS}

Ambiguous: Use when the emotional meaning is weak, generic, or cannot be reliably linked to the video scenario.
Examples: "I love it", "Amazing", "Wow", "So good"

{BORDERLINE}

Instructions:
• Use only explicitly available evidence.
• Do not assume missing context.
• Focus only on non-clinical emotional reactions.
• Preserve uncertainty — do not default to None or Empathy when evidence is weak.
• If the emotional meaning cannot be reliably interpreted, return "Ambiguous".
• If the emotional meaning is clear, classify the emotion using the predefined labels.
• Provide a concise explanation for the result.
If a single label clearly explains the emotional reaction, return only that label.
Only return a secondary label when two categories are strongly supported, neither is clearly preferred, and both are reasonable.
Otherwise, return a single label.

Input Comment: {{comment}}

Respond ONLY in the following format:

Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None / Ambiguous>
Secondary Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None / Ambiguous>

Explanation:
"""

P7 = f"""{header("P7", "No-RAG Emotion Classification (Explicit branch)", [
    "Stage 1 — Prompt Role Separation (Explicit branch; no intermediate ambiguity check)",
    "Stage 2 — Video-Anchored Emotion Definitions & Empathy vs Frustration Guidance",
    "Stage 3 — Borderline Decision Rules (six codified rules)",
])}

You are an emotion classification system for non-clinical emotional reactions in user comments on a social experiment video.

Your task is to classify the commenter's emotional reaction based on what they saw in the video — how the social experiment made them feel or how they respond to people and events in the video.

Use ONLY:
1. The explicitly available information in the comment.
2. The provided video scenario.

Do NOT infer hidden intentions, unstated emotions, or external context beyond the provided scenario.

Video Scenario:
{SCENARIO}

Predefined non-clinical emotional state labels:
• Shock
• Disbelief
• Anger
• Frustration
• Empathy
• None

{VIDEO_DEFS}

{BORDERLINE}

Instructions:
• Use only explicitly available evidence.
• Do not assume missing context.
• Focus only on non-clinical emotional reactions.
• If multiple emotions appear, choose the dominant explicitly expressed emotion.
• Provide a concise explanation for the classification result.
If a single label clearly explains the emotional reaction, return only that label.
Only return a secondary label when two categories are strongly supported, neither is clearly preferred, and both are reasonable.
Otherwise, return a single label.

Input Comment: {{comment}}

Respond ONLY in the following format:
Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None>
Secondary Emotion Label: <Shock / Disbelief / Anger / Frustration / Empathy / None>
Explanation:
"""

PROMPTS = {
    "P1_prompt.txt": P1,
    "P2_prompt.txt": P2,
    "P3_prompt.txt": P3,
    "P4_prompt.txt": P4,
    "P5_prompt.txt": P5,
    "P6_prompt.txt": P6,
    "P7_prompt.txt": P7,
}


def main() -> None:
    for fname, content in PROMPTS.items():
        (V1 / fname).write_text(content, encoding="utf-8")
        print(f"Wrote {V1 / fname}")
    print("Canonical P1–P7 updated. Run build_experiment_video_assets.py to sync per-video copies.")


if __name__ == "__main__":
    main()
