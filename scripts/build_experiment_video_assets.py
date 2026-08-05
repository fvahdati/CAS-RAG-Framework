#!/usr/bin/env python3
"""Generate per-video metadata, prototypes, and prompts under resources/datasets/Experiments-data."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resources" / "datasets" / "Experiments-data"
PROMPT_TEMPLATE_DIR = ROOT / "resources" / "prompts" / "v1"

VIDEOS = [
    {
        "video_number": 1,
        "slug": "video_01_bullying_bystander_bus_stop",
        "dataset_file": "comments_bullying",
        "display_name": "Bullying Bystander — Bus Stop",
        "youtube_video_id": "Vo7kl9RsWBo",
        "topic_theme": "Teenagers bullying prevention",
        "evaluation_label": "Bullying",
        "short_scenario": "A social experiment shows how bystanders respond when a teenage girl is verbally bullied in public.",
        "full_scenario": (
            "This staged social experiment takes place at a public bus stop where a teenage girl is sitting alone reading a book. "
            "Two individuals begin verbally bullying her, mocking her appearance, personality, and social life. "
            "The girl becomes visibly uncomfortable but does not respond. Initially, bystanders watch without intervening, "
            "but as the situation escalates, several people step in—some confront the bullies directly, while others support "
            "the victim by offering comfort and inclusion. The experiment highlights different types of bystander behavior and "
            "emphasizes the importance of standing up against bullying and supporting vulnerable individuals."
        ),
        "prototypes": {
            "explicit": [
                "I love how strangers stood up for that girl at the bus stop",
                "This restored my faith in humanity—people actually intervened",
                "Bullying is never okay and seeing adults step in was beautiful",
                "So proud of everyone who confronted the bullies",
                "Wholesome experiment showing real courage in public",
            ],
            "vague": [
                "Not sure if this is inspiring or just another staged WWYD clip",
                "Mixed feelings about hidden-camera bullying experiments",
                "Hard to tell how much of this was real versus produced",
                "I don't know how to feel about filming victims in public",
                "Interesting but makes me uneasy about consent and editing",
            ],
            "context_dependent": [
                "The man with the harmonica made my heart melt",
                "The last guy though... that reaction hit different",
                "When that happened at the end I started crying",
                "That part where they stepped in was the best moment",
                "Who was the person playing harmonica—absolute legend",
            ],
        },
    },
    {
        "video_number": 2,
        "slug": "video_02_education_tutoring_diner",
        "dataset_file": "comments_education",
        "display_name": "Education — Tutoring Social Experiment (Diner)",
        "youtube_video_id": "lqevHrpvAx8",
        "topic_theme": "Educational videos for adolescents",
        "evaluation_label": "Education",
        "short_scenario": "A hidden camera experiment observes how children react when a tutor verbally mistreats a struggling student.",
        "full_scenario": (
            "A hidden camera experiment is set up in a public diner where a child actor struggles with basic math while a tutor "
            "becomes increasingly rude and verbally harsh. The situation is designed to test how other children nearby will respond. "
            "Some hesitate at first, but several step in to help the struggling student, offering both academic support and emotional "
            "encouragement. The experiment reveals that even young children can recognise unfair treatment and act with empathy, "
            "highlighting the importance of kindness, confidence, and standing up against bullying behaviour in educational settings."
        ),
        "prototypes": {
            "explicit": [
                "The way those kids jumped in proves empathy can beat a bad adult example every time.",
                "This is exactly why kindness and confidence matter more than any worksheet.",
                "Restores a little faith—kids noticing unfairness and refusing to shrug it off.",
                "Great reminder that bullying in learning spaces is never okay, no matter the adult's excuse.",
                "Wholesome watch; we need way more classrooms that normalise standing up for someone struggling.",
            ],
            "vague": [
                "Cute message but I'm still uneasy about hidden-camera setups in public places.",
                "Mixed feelings—is this stitched like reality TV or genuinely spontaneous?",
                "Inspiring edits, though I wish we knew consent and editing context upfront.",
                "Not sure the lesson needs a diner stunt; feels performative even if the vibe is sweet.",
                "Hard to tell what's heartfelt vs what producers nudged; still liked the takeaway.",
            ],
            "context_dependent": [
                "Aubrey calmly calling out that tutor felt more composed than half the adults online.",
                "When Harper walked over with the milkshake energy I melted—pure emotional support.",
                "Everybody's a hero—that kid's line was the gut-punch moment of the whole bit.",
                "The waitress clocking oh right he's acting cracked the fourth wall harder than WWYD usually does.",
                "Rough tutor banging patience at the booth while strangers' kids leaned in—is peak WWYD diner tension.",
            ],
        },
    },
    {
        "video_number": 3,
        "slug": "video_03_mental_health_suicide_bridge",
        "dataset_file": "comments_Suicide",
        "display_name": "Mental Health — Bridge Distress Experiment",
        "youtube_video_id": "Gwo_O0SFwak",
        "topic_theme": "Teenage depression / mental-health-related adolescent themes",
        "evaluation_label": "Suicide",
        "short_scenario": "A staged social experiment shows how strangers react when a distressed young man appears to be attempting suicide on a bridge.",
        "full_scenario": (
            "This social experiment presents a staged scenario in which a young man appears emotionally distressed and attempts to jump off a bridge. "
            "Hidden cameras capture how bystanders react in real time. Several strangers intervene, showing empathy and urgency—some physically stop him, "
            "while others offer emotional support, reassurance, and companionship. They encourage him to talk, remind him of his value, and try to de-escalate "
            "the situation. The experiment highlights the importance of human compassion and the role of bystanders in preventing tragedy, emphasizing themes "
            "of depression, loneliness, and human connection."
        ),
        "prototypes": {
            "explicit": [
                "I'm crying—the kindness in this video gives me so much hope",
                "Small acts of compassion like this can really save someone's life",
                "This restored my faith in humanity after a rough week",
                "We need more people who stop and actually care in public",
                "So powerful to see strangers offer real support on a bridge",
            ],
            "vague": [
                "I don't know if I'd know what to say in that situation myself",
                "This is hard to watch but I think it's important content",
                "Mixed emotions—inspiring but also deeply sad at the same time",
                "Not sure if filming someone in distress is ethical even for awareness",
                "I'm conflicted about bystander experiments involving mental health",
            ],
            "context_dependent": [
                "The woman who sat down next to him made my whole week",
                "When he asked are you okay quietly I completely lost it",
                "That guy in the hoodie who didn't walk away was a hero",
                "The moment someone offered to sit with them hit different",
                "3:32 the fact that she was crying shows how big her heart is",
            ],
        },
    },
    {
        "video_number": 4,
        "slug": "video_04_teen_safety_predator_cafe",
        "dataset_file": "comments-TeenSafety",
        "display_name": "Teen Safety — Predator Luring (Café)",
        "youtube_video_id": "YdqRLuYkYDk",
        "topic_theme": "Teen-safety / adolescent mental-health awareness",
        "evaluation_label": "TeenSafety",
        "short_scenario": "Bystanders react when a teenage girl is lured by an older man into a potentially dangerous situation.",
        "full_scenario": (
            "This social experiment shows a 15-year-old girl being approached by an older man who attempts to lure her into a private setting "
            "under the promise of helping her music career. The interaction unfolds in a public café, where hidden cameras record how bystanders react. "
            "Some people intervene, warning the girl and confronting the man, while others hesitate. The situation explores teenage vulnerability, "
            "manipulation tactics, and the importance of awareness and intervention. It highlights how emotional naivety, trust, and social pressure "
            "can place teenagers in risky situations, linking closely to mental and emotional well-being."
        ),
        "prototypes": {
            "explicit": [
                "Thank god those women stepped in—that predator needed to be stopped",
                "Every parent should show this to their teenage kids",
                "So angry at the man but grateful people intervened",
                "This is why we teach girls to trust their instincts in public",
                "Heartwarming to see strangers protect her in the café",
            ],
            "vague": [
                "Not sure how real this setup is but it's unsettling",
                "Mixed feelings about staging predatory behaviour for views",
                "I don't know if I'd react that fast in a café",
                "Hard to watch regardless of whether it's an experiment",
                "Interesting social experiment but ethically complicated",
            ],
            "context_dependent": [
                "When the first girl with her dad started crying, so did I",
                "Imagine someone hiring you because you look like a creepy old man",
                "Megan said he seems nice—so did Ted Bundy",
                "Who was the actor playing the predator—don't blame him",
                "4:00 that first girl's panic is so valid and made me sob",
            ],
        },
    },
    {
        "video_number": 5,
        "slug": "video_05_teen_pregnancy_gabby_experiment",
        "dataset_file": "comments_teenPregnancy",
        "display_name": "Teen Pregnancy Prevention — Gabby's School Project",
        "youtube_video_id": "8ddhPrll65Y",
        "topic_theme": "Teen-pregnancy prevention",
        "evaluation_label": "TeenPregnancy",
        "short_scenario": "A high school student fakes a pregnancy for months as a social experiment to reveal judgment toward teen mothers.",
        "full_scenario": (
            "A top-performing high school student named Gabby conducts a bold social experiment by pretending to be pregnant for over six months. "
            "With only a few people aware of the truth, she convinces her classmates, teachers, and community that she is expecting a child. "
            "As the fake pregnancy progresses, she carefully documents how people's behavior toward her changes—friends distance themselves, "
            "teachers lower their expectations, and rumors spread. Despite her strong academic background and ambitions, she is reduced to a stereotype. "
            "At a school assembly, she reveals the truth, exposing how society unfairly judges and labels teenage mothers. Her project becomes a powerful "
            "statement about bias, perception, and social judgment."
        ),
        "prototypes": {
            "explicit": [
                "Sacrificing your last year of school to make such an incredible statement—wow",
                "This project exposed how cruel people are to pregnant teens",
                "Gabby is brilliant for showing real-world bias so clearly",
                "Every school should watch this assembly reveal",
                "I'm furious at how her friends treated her once they thought she was pregnant",
            ],
            "vague": [
                "Not sure how I feel about faking pregnancy for a school project",
                "Mixed emotions—it taught a lesson but also feels intense",
                "I don't know if this method was fair to her classmates",
                "Powerful message but ethically complicated to deceive everyone",
                "Important topic but hard to judge the experiment design",
            ],
            "context_dependent": [
                "When she revealed it at the assembly I lost it",
                "Gabby Rodriguez is an icon for doing this project",
                "That moment her friends realised it was fake—chills",
                "The teacher who lowered expectations should be ashamed",
                "Who else paused when she showed the six-month timeline",
            ],
        },
    },
    {
        "video_number": 6,
        "slug": "video_06_drugs_prevention_bystander",
        "dataset_file": "comments_teenagers drugs",
        "display_name": "Drug Prevention — Normalising Use (Public Setting)",
        "youtube_video_id": "0yve1ezfeUU",
        "topic_theme": "Drug-prevention context for adolescents",
        "evaluation_label": "Teenagers-Drugs",
        "short_scenario": "A social experiment tests whether people intervene when an adult encourages a teenager to normalise drug use.",
        "full_scenario": (
            "In this staged scenario, a teenager is confronted by his mother about drug use in a public setting. When the mother leaves temporarily, "
            "an adult actor intervenes and begins encouraging the teen to dismiss his mother's concerns, normalising drug use and even offering access to drugs. "
            "Hidden cameras capture how bystanders react—some remain passive due to uncertainty about interfering in parenting, while others step in to warn the mother "
            "or challenge the inappropriate influence. The experiment highlights peer pressure, normalization of risky behaviors, and the critical role of bystanders "
            "in protecting young individuals."
        ),
        "prototypes": {
            "explicit": [
                "Thank god someone told the mom what that stranger was saying",
                "No teenager should be told drugs are fine—angry this happened",
                "Proud of the people who stepped in against that adult",
                "This is exactly why bystanders matter for teen safety",
                "Disgusted by the actor normalising drug use to a kid",
            ],
            "vague": [
                "Not sure if I'd interfere in someone else's parenting in public",
                "Mixed feelings about staging drug pressure for a video",
                "Hard to watch but maybe it raises awareness",
                "I don't know if this experiment went too far",
                "Uncomfortable content regardless of the message",
            ],
            "context_dependent": [
                "When the mom came back and heard what he said—I cheered",
                "No way they did this at a place called HIGH country",
                "That stranger offering drugs should be arrested",
                "The bystander who warned the mother is the real MVP",
                "Who was the adult actor—don't confuse him with a real dealer",
            ],
        },
    },
]


def _read_template(name: str) -> str:
    return (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _inject_scenario(template: str, scenario: str) -> str:
    """Replace scenario blocks while preserving refinement headers and shared rule sections."""
    lines = template.splitlines()
    out: list[str] = []
    skip_until_marker = False
    scenario_labels = ("Video Scenario:", "Short Video Scenario:")
    stop_prefixes = (
        "Predefined",
        "Stage 1",
        "Definitions",
        "Explicit Emotional",
        "Ambiguous Emotional",
        "A reference should",
        "Contrastive Examples",
        "Instructions:",
        "Label Definitions",
        "Borderline Decision",
        "Rule 1",
        "You are a semantic similarity",
    )
    for line in lines:
        if any(line.startswith(label) for label in scenario_labels):
            label = "Short Video Scenario:" if line.startswith("Short Video Scenario:") else "Video Scenario:"
            out.append(label)
            out.append(scenario)
            skip_until_marker = True
            continue
        if skip_until_marker:
            if line.strip() == "":
                continue
            if any(line.startswith(prefix) for prefix in stop_prefixes):
                skip_until_marker = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def _build_p5(scenario: str, prototypes: dict) -> str:
    """Inject per-video scenario and prototype banks into canonical P5 template."""
    template = _inject_scenario(_read_template("P5_prompt.txt"), scenario)
    exp_block = "\n".join(f'{i + 1}. "{p}"' for i, p in enumerate(prototypes["explicit"]))
    vague_block = "\n".join(f'{i + 1}. "{p}"' for i, p in enumerate(prototypes["vague"]))
    if "Explicit Emotional Reaction Prototypes" not in template:
        return template
    exp_start = template.index("Explicit Emotional Reaction Prototypes")
    amb_start = template.index("Ambiguous Emotional Reaction Prototypes")
    instr_start = template.index("Instructions:")
    head = template[:exp_start]
    tail = template[instr_start:]
    mid = (
        "Explicit Emotional Reaction Prototypes (Stage 1 — self-contained opinions & scenario-identifiable reactions):\n\n"
        f"{exp_block}\n\n"
        "Ambiguous Emotional Reaction Prototypes (Stage 1 — weak signal, authenticity doubt, generic praise):\n\n"
        f"{vague_block}\n\n"
    )
    return head + mid + tail


def write_video_package(video: dict) -> None:
    base = OUT / video["slug"]
    (base / "prompts").mkdir(parents=True, exist_ok=True)

    (base / "scenario_full.txt").write_text(video["full_scenario"], encoding="utf-8")
    (base / "scenario_short.txt").write_text(video["short_scenario"], encoding="utf-8")

    prototypes_path = base / "prototypes_semantic.json"
    prototypes_path.write_text(
        json.dumps(
            {
                "video_slug": video["slug"],
                "youtube_video_id": video["youtube_video_id"],
                "explicit": video["prototypes"]["explicit"],
                "vague": video["prototypes"]["vague"],
                "context_dependent": video["prototypes"]["context_dependent"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "video_number": video["video_number"],
        "slug": video["slug"],
        "display_name": video["display_name"],
        "youtube_video_id": video["youtube_video_id"],
        "topic_theme": video["topic_theme"],
        "evaluation_label": video["evaluation_label"],
        "dataset_source_file": video["dataset_file"],
        "comments_per_dataset": 100,
        "scenario_short": video["short_scenario"],
        "scenario_full": video["full_scenario"],
        "paths": {
            "scenario_full": f"{video['slug']}/scenario_full.txt",
            "scenario_short": f"{video['slug']}/scenario_short.txt",
            "prototypes": f"{video['slug']}/prototypes_semantic.json",
            "prompts_dir": f"{video['slug']}/prompts/",
            "script_corpus": f"{video['slug']}/script_corpus.txt",
        },
        "cas_rag_layer1_categories": ["Explicit", "Ambiguous", "Context-Dependent"],
        "emotion_labels": ["Shock", "Disbelief", "Anger", "Frustration", "Empathy", "None"],
        "human_verification": "required",
    }
    (base / "video_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    (base / "script_corpus.txt").write_text(
        f"# Placeholder retrieval corpus for {video['display_name']}\n"
        f"# Replace with video transcript chunks (YouTube captions / ASR).\n"
        f"# YouTube ID: {video['youtube_video_id']}\n\n"
        f"Scene summary: {video['short_scenario']}\n",
        encoding="utf-8",
    )

    scenario = video["full_scenario"]
    mapping = {
        "P1_unknown_reference_detection.txt": _inject_scenario(_read_template("P1_prompt.txt"), scenario),
        "P2_rag_emotion_classification.txt": _inject_scenario(_read_template("P2_prompt.txt"), scenario),
        "P3_no_rag_emotion_classification.txt": _inject_scenario(_read_template("P3_prompt.txt"), scenario),
        "P4_semantic_ambiguity_verification.txt": _inject_scenario(_read_template("P4_prompt.txt"), scenario),
        "P5_semantic_similarity_explicit_vague.txt": _build_p5(scenario, video["prototypes"]),
        "P6_semantic_ambiguity_aware_emotion.txt": _inject_scenario(_read_template("P6_prompt.txt"), scenario),
        "P7_no_rag_emotion_without_vagueness_check.txt": _inject_scenario(_read_template("P7_prompt.txt"), scenario),
    }
    for fname, content in mapping.items():
        (base / "prompts" / fname).write_text(content, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"videos": [], "total_comments": 600, "comments_per_video": 100}
    for v in VIDEOS:
        write_video_package(v)
        manifest["videos"].append(
            {
                "video_number": v["video_number"],
                "slug": v["slug"],
                "display_name": v["display_name"],
                "youtube_video_id": v["youtube_video_id"],
                "evaluation_label": v["evaluation_label"],
                "metadata_file": f"{v['slug']}/video_metadata.json",
                "prototypes_file": f"{v['slug']}/prototypes_semantic.json",
                "prompts_directory": f"{v['slug']}/prompts/",
            }
        )
    (OUT / "experiments_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(VIDEOS)} video packages to {OUT}")


if __name__ == "__main__":
    main()
