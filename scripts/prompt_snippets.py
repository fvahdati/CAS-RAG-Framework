"""Per-video P1/P2 scenario-specific prompt snippets (Stage 1–3 refinements)."""

from __future__ import annotations

# Shared Stage 3 borderline rules — logic is generalized; only examples vary per video.
_BORDERLINE_HEADER = """Borderline Decision Rules (Stage 3 — codified borderline cases):

Rule 1 — Personal Abuse/Mistreatment → Anger
If a commenter describes being abused, bullied, humiliated, manipulated, judged, or physically or verbally mistreated, classify as Anger when the abusive or unfair behaviour is the dominant focus — even if the commenter never explicitly says they were angry.
Example: "{rule1_example}" → Anger

Rule 2 — Shared Experience ≠ Empathy
Sharing a similar personal experience alone does NOT indicate Empathy.
Empathy requires explicit compassion, concern, encouragement, sympathy, support, or admiration toward another person in the video.
Autobiographical stories should be classified by their dominant emotional reaction, not automatically as Empathy.

Rule 3 — Advice and Constructive Support
Comments offering advice should not be classified solely because advice is present.
• If the advice reflects concern, support, patience, or advocacy for a vulnerable person in response to events in the video → Empathy.
• If the advice is a neutral opinion without a clear emotional reaction toward people or events in the video → None.
{rule3_note}

Rule 4 — Questioning Reality → Disbelief
Comments questioning whether the experiment, participants, or setting are real, believable, staged, or authentic → Disbelief.
Examples: "This looks fake." / "No way this happened." / "That person is definitely an actor."

Rule 5 — Generic Praise
Generic praise ({rule5_terms}) → Empathy ONLY when it clearly refers to kindness, compassion, encouragement, support, or intervention shown in the experiment.
Otherwise → None or Ambiguous (when applicable).

Rule 6 — Video Reaction Priority
When a comment contains both a personal story and a reaction to the video, classify according to the commenter's emotional reaction toward the events shown in the social experiment.
Do NOT classify solely based on the emotion in the personal story.
The personal story is supporting context only.

Praise vs Empathy (Stage 2 follow-up):
Words such as sweet, cute, or beautiful directed at behaviour are NOT automatically Empathy unless they express compassion toward a person in distress or admiration for kindness shown in the experiment."""


def _borderline(rule1_example: str, rule3_note: str, rule5_terms: str) -> str:
    return _BORDERLINE_HEADER.format(
        rule1_example=rule1_example,
        rule3_note=rule3_note,
        rule5_terms=rule5_terms,
    )


P1_SNIPPETS: dict[str, dict[str, str]] = {
    "video_01_bullying_bystander_bus_stop": {
        "named_characters_hint": " (e.g., the man with the harmonica, that last guy)",
        "resolved_criteria": """• the comment clearly refers to the victim, the bullies, intervening bystanders, or the bus-stop bullying experiment,
• the comment is a general hypothetical reaction to public bullying without naming an unresolved person,
• or identifiable scenario entities ("that girl", "the bullies", "those bystanders") resolve all referents.""",
        "contrastive_examples": """Comment: "She handled that really well."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "Those bystanders were so brave."
Unknown Reference: No
Explanation: "Those bystanders" refers to people who stepped in — identifiable from the scenario.

Comment: "If people talked to me like that at a bus stop I'd be devastated."
Unknown Reference: No
Explanation: Refers to the bullying behaviour from the scenario; no unresolved entity.

Comment: "When the man with the harmonica stepped in I melted."
Unknown Reference: Yes
Explanation: Requires script context to identify who the man with the harmonica is and which moment is referenced.

Comment: "That last guy though—that reaction hit different."
Unknown Reference: Yes
Explanation: Refers to a specific bystander or moment not identifiable from the scenario summary alone.""",
    },
    "video_02_education_tutoring_diner": {
        "named_characters_hint": " (e.g., Harper, Aubrey)",
        "resolved_criteria": """• the comment clearly refers to the tutor, the struggling child, the helping children, or the diner experiment,
• the comment is a general hypothetical reaction to the tutor's harsh behaviour without naming an unresolved person,
• or identifiable scenario entities ("those kids", "the tutor", "that child") resolve all referents.""",
        "contrastive_examples": """Comment: "She handled that really well."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "Those kids were so brave."
Unknown Reference: No
Explanation: "Those kids" refers to nearby children who helped — identifiable from the scenario.

Comment: "If I had a tutor that harsh I would feel so stupid."
Unknown Reference: No
Explanation: Refers to the tutor's harsh behaviour from the scenario; no unresolved entity.

Comment: "When Harper walked over I melted."
Unknown Reference: Yes
Explanation: Requires script context to identify who Harper is and which moment is referenced.

Comment: "Aubrey calmly calling out that tutor felt composed."
Unknown Reference: Yes
Explanation: Named character Aubrey is not identifiable from the scenario summary alone.""",
    },
    "video_03_mental_health_suicide_bridge": {
        "named_characters_hint": " (e.g., the woman who sat down, the guy in the hoodie)",
        "resolved_criteria": """• the comment clearly refers to the distressed young man, intervening strangers, or the bridge suicide-prevention experiment,
• the comment is a general hypothetical reaction to someone in crisis without naming an unresolved person,
• or identifiable scenario entities ("that man", "the strangers", "people on the bridge") resolve all referents.""",
        "contrastive_examples": """Comment: "She handled that really well."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "The kindness of those strangers was beautiful."
Unknown Reference: No
Explanation: Refers to bystanders who intervened — identifiable from the scenario.

Comment: "If I saw someone on a bridge like that I'd stop too."
Unknown Reference: No
Explanation: Refers to the crisis scenario from the video; no unresolved entity.

Comment: "When the woman who sat down next to him spoke, I cried."
Unknown Reference: Yes
Explanation: Requires script context to identify which woman and which specific moment is referenced.

Comment: "That guy in the hoodie who didn't walk away was a hero."
Unknown Reference: Yes
Explanation: Refers to a specific bystander not identifiable from the scenario summary alone.""",
    },
    "video_04_teen_safety_predator_cafe": {
        "named_characters_hint": " (e.g., Megan, the actor playing the predator)",
        "resolved_criteria": """• the comment clearly refers to the teenage girl, the older man, intervening bystanders, or the café predator-luring experiment,
• the comment is a general hypothetical reaction to grooming or manipulation without naming an unresolved person,
• or identifiable scenario entities ("that girl", "the man", "those women") resolve all referents.""",
        "contrastive_examples": """Comment: "She was so naive."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "Thank god those women stepped in."
Unknown Reference: No
Explanation: Refers to bystanders who intervened — identifiable from the scenario.

Comment: "If a stranger tried to lure me like that I'd run."
Unknown Reference: No
Explanation: Refers to the predator-luring scenario; no unresolved entity.

Comment: "When Megan said he seems nice I lost it."
Unknown Reference: Yes
Explanation: Requires script context to identify who Megan is and which moment is referenced.

Comment: "Imagine someone hiring you because you look like a creepy old man."
Unknown Reference: Yes
Explanation: Refers to production or casting details not inferable from the scenario summary alone.""",
    },
    "video_05_teen_pregnancy_gabby_experiment": {
        "named_characters_hint": " (e.g., Gabby, Gabby Rodriguez)",
        "resolved_criteria": """• the comment clearly refers to Gabby, classmates, teachers, the school assembly reveal, or the teen-pregnancy experiment,
• the comment is a general hypothetical reaction to judgment toward teen mothers without naming an unresolved person,
• or identifiable scenario entities ("she", "her friends", "the assembly") resolve all referents.""",
        "contrastive_examples": """Comment: "She handled that really well."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "Gabby is brilliant for exposing how cruel people are."
Unknown Reference: No
Explanation: Refers to Gabby — the student conducting the experiment, identifiable from the scenario.

Comment: "If my friends treated me like that when I was pregnant I'd be crushed."
Unknown Reference: No
Explanation: Refers to social judgment in the experiment; no unresolved entity.

Comment: "When she revealed it at the assembly I lost it."
Unknown Reference: Yes
Explanation: Requires script context to identify the specific assembly moment referenced.

Comment: "Gabby Rodriguez is an icon for doing this project."
Unknown Reference: Yes
Explanation: Full name and project details require script context beyond the scenario summary alone.""",
    },
    "video_06_drugs_prevention_bystander": {
        "named_characters_hint": " (e.g., the adult actor, the bystander who warned the mother)",
        "resolved_criteria": """• the comment clearly refers to the teenager, the mother, the stranger encouraging drug use, intervening bystanders, or the drug-prevention experiment,
• the comment is a general hypothetical reaction to drug pressure without naming an unresolved person,
• or identifiable scenario entities ("that kid", "the mom", "the stranger") resolve all referents.""",
        "contrastive_examples": """Comment: "She did the right thing."
Unknown Reference: Yes
Explanation: The identity of "she" is unclear from the scenario alone.

Comment: "Thank god someone told the mom what that stranger was saying."
Unknown Reference: No
Explanation: Refers to the mother and intervening bystander — identifiable from the scenario.

Comment: "No teenager should be told drugs are fine."
Unknown Reference: No
Explanation: Refers to the drug-normalisation scenario; no unresolved entity.

Comment: "When the mom came back and heard what he said—I cheered."
Unknown Reference: Yes
Explanation: Requires script context to identify the specific moment when the mother returned.

Comment: "Who was the adult actor—don't confuse him with a real dealer."
Unknown Reference: Yes
Explanation: Refers to casting or production details not inferable from the scenario summary alone.""",
    },
}

P2_SNIPPETS: dict[str, dict[str, str]] = {
    "video_01_bullying_bystander_bus_stop": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at the bullying shown in the video.
Example: "I can't believe they mocked her like that at a bus stop."

Disbelief: Difficulty accepting that the bullying, participants, or setup is real or believable.
Example: "No way people would bully a girl just for reading a book in public."

Anger: Moral outrage at bullying, humiliation, or unfair treatment shown in the video — including personal stories where abusive behaviour is the dominant focus.
Example: "Verbal bullying at a bus stop is absolutely disgusting."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward the bullying or passive bystander behaviour.
Example: "Watching bystanders hesitate before stepping in was so frustrating."

Empathy: Compassion, support, admiration, or kindness expressed toward the victim or the bystanders who intervened — NOT merely sharing a similar personal experience.
Example: "I love how strangers stood up for that girl at the bus stop."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "WWYD always films bullying scenarios in public."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises intervening bystanders, admires kindness, or expresses compassion toward the victim — even if bullying is mentioned hypothetically.
• Choose Frustration when annoyance, helplessness, or dissatisfaction with the bullying or bystander inaction is the dominant reaction.
Example (Empathy): "If I were her I'd be terrified — bless everyone who stepped in.\"""",
        "borderline_rules": _borderline(
            rule1_example="Kids at my school used to mock me like that at the bus stop",
            rule3_note="Do not infer Empathy solely because the commenter proposes how bystanders should respond to bullying.",
            rule5_terms='"so brave", "great people", "wonderful"',
        ),
    },
    "video_02_education_tutoring_diner": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at what happened in the video.
Example: "I can't believe he humiliated that child over basic math."

Disbelief: Difficulty accepting that the tutor's behavior, participants, or setup is real or believable.
Example: "No way a real tutor would talk to a kid like that."

Anger: Moral outrage at unfair treatment, abuse, bullying, or humiliation shown in the video — including personal stories where abusive behaviour is the dominant focus.
Example: "That tutor should be fired for tearing down a struggling student."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward the situation or ineffective teaching.
Example: "Watching a kid get torn down like that is so frustrating."

Empathy: Compassion, support, admiration, or kindness expressed toward the struggling child or the children who helped — NOT merely sharing a similar personal experience.
Example: "Those kids were so sweet to step in and help her."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "WWYD always films in diners."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises helping children, admires kindness, or expresses compassion toward the struggling child — even if harsh treatment is mentioned hypothetically.
• Choose Frustration when annoyance, helplessness, or dissatisfaction with the situation or teaching is the dominant reaction.
Example (Empathy): "If I had a tutor that harsh I would feel so stupid — bless those kids for helping.\"""",
        "borderline_rules": _borderline(
            rule1_example="My tutor hit me for getting the answer wrong",
            rule3_note="Do not infer Empathy solely because the commenter proposes a better teaching method.",
            rule5_terms='"great kids", "so sweet", "wonderful"',
        ),
    },
    "video_03_mental_health_suicide_bridge": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at the distress or suicide attempt shown in the video.
Example: "I can't believe someone would try to jump off a bridge in public."

Disbelief: Difficulty accepting that the scenario, participants, or setup is real or believable.
Example: "No way this wasn't staged — who films someone on a bridge like that?"

Anger: Moral outrage at neglect, stigma, or harmful treatment related to mental health — including personal stories where mistreatment or isolation is the dominant focus.
Example: "Society fails people with depression every single day."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward the crisis or inadequate support.
Example: "It breaks my heart how hard it is to know what to say in that moment."

Empathy: Compassion, support, admiration, or kindness expressed toward the distressed person or the strangers who intervened — NOT merely sharing a similar personal experience.
Example: "The kindness in this video gives me so much hope."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "Mental health awareness videos are everywhere now."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises intervening strangers, admires compassion, or expresses support toward the distressed person — even if crisis is mentioned hypothetically.
• Choose Frustration when helplessness, sadness without clear compassion toward others, or dissatisfaction with the situation is the dominant reaction.
Example (Empathy): "If I saw someone on a bridge like that I'd sit with them too — those strangers are heroes.\"""",
        "borderline_rules": _borderline(
            rule1_example="I've been that low on a bridge before and nobody stopped",
            rule3_note="Do not infer Empathy solely because the commenter offers advice about how to help someone in crisis.",
            rule5_terms='"so kind", "beautiful", "heroes"',
        ),
    },
    "video_04_teen_safety_predator_cafe": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at the predatory behaviour shown in the video.
Example: "I can't believe an older man tried to lure a 15-year-old in a café."

Disbelief: Difficulty accepting that the scenario, participants, or setup is real or believable.
Example: "No way a predator would be that bold in a public café."

Anger: Moral outrage at predatory behaviour, grooming, or manipulation — including personal stories where exploitation is the dominant focus.
Example: "That predator needed to be stopped immediately."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward the danger or bystander hesitation.
Example: "Watching people hesitate to intervene was so frustrating."

Empathy: Compassion, support, admiration, or kindness expressed toward the teenage girl or the bystanders who protected her — NOT merely sharing a similar personal experience.
Example: "Heartwarming to see strangers protect her in the café."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "Teen safety videos get a lot of views."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises intervening bystanders, admires protection of the girl, or expresses concern for her safety — even if predatory behaviour is mentioned hypothetically.
• Choose Frustration when annoyance, helplessness, or dissatisfaction with the danger or hesitation is the dominant reaction.
Example (Empathy): "If a stranger tried to lure me like that I'd hope someone would step in — thank god those women did.\"""",
        "borderline_rules": _borderline(
            rule1_example="An older man tried to groom me the same way when I was fifteen",
            rule3_note="Do not infer Empathy solely because the commenter offers safety advice for teenagers.",
            rule5_terms='"thank god", "brave", "heroes"',
        ),
    },
    "video_05_teen_pregnancy_gabby_experiment": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at the judgment or social experiment reveal shown in the video.
Example: "I can't believe her friends treated her so differently once they thought she was pregnant."

Disbelief: Difficulty accepting that the experiment, participants, or setup is real or believable.
Example: "No way she faked a pregnancy for six months without anyone finding out."

Anger: Moral outrage at judgment, stigma, or unfair treatment of teen mothers — including personal stories where social rejection is the dominant focus.
Example: "I'm furious at how her friends treated her once they thought she was pregnant."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward bias or lowered expectations.
Example: "Watching teachers lower their expectations just because of a pregnancy was so frustrating."

Empathy: Compassion, support, admiration, or kindness expressed toward Gabby or others judged unfairly — NOT merely sharing a similar personal experience.
Example: "Gabby is brilliant for showing real-world bias so clearly."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "School projects about social experiments are common now."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises Gabby's courage, admires the reveal, or expresses compassion toward judged teens — even if stigma is mentioned hypothetically.
• Choose Frustration when annoyance, helplessness, or dissatisfaction with judgment or bias is the dominant reaction.
Example (Empathy): "If my friends treated me like that when I was pregnant I'd be crushed — Gabby's project was powerful.\"""",
        "borderline_rules": _borderline(
            rule1_example="My friends stopped talking to me when they thought I was pregnant",
            rule3_note="Do not infer Empathy solely because the commenter offers opinions about how schools should treat pregnant teens.",
            rule5_terms='"brilliant", "icon", "powerful"',
        ),
    },
    "video_06_drugs_prevention_bystander": {
        "emotion_definitions": """Label Definitions (video-anchored — Stage 2):
Classify the dominant reaction a viewer expresses after watching this social experiment.

Shock: Surprise or emotional disturbance at an adult encouraging a teenager to use drugs.
Example: "I can't believe a stranger offered drugs to a kid while his mom was away."

Disbelief: Difficulty accepting that the scenario, participants, or setup is real or believable.
Example: "No way someone would openly normalise drug use to a teenager in public."

Anger: Moral outrage at drug normalisation, peer pressure, or endangering a minor — including personal stories where drug pressure is the dominant focus.
Example: "Disgusted by the actor normalising drug use to a kid."

Frustration: Annoyance, helplessness, or emotional dissatisfaction as the dominant reaction toward the danger or bystander hesitation.
Example: "Watching adults hesitate to interfere in someone else's parenting was frustrating."

Empathy: Compassion, support, admiration, or kindness expressed toward the teenager, the mother, or bystanders who intervened — NOT merely sharing a similar personal experience.
Example: "Proud of the people who stepped in against that adult."

None: Neutral factual comments, jokes without clear emotion, generic reactions without a reliable link to the video, or reactions outside the five emotion categories.
Example: "Drug prevention videos are important content."

Empathy vs Frustration (important — Stage 2):
• Choose Empathy when the comment praises intervening bystanders, admires protection of the teen, or expresses concern for the family — even if drug pressure is mentioned hypothetically.
• Choose Frustration when annoyance, helplessness, or dissatisfaction with the danger or hesitation is the dominant reaction.
Example (Empathy): "If a stranger offered my kid drugs I'd want someone to warn me — that bystander is the real MVP.\"""",
        "borderline_rules": _borderline(
            rule1_example="A stranger once offered me drugs when my parents weren't around",
            rule3_note="Do not infer Empathy solely because the commenter offers parenting or drug-prevention advice.",
            rule5_terms='"MVP", "hero", "proud"',
        ),
    },
}


def get_p1_snippets(slug: str) -> dict[str, str]:
    return P1_SNIPPETS[slug]


def get_p2_snippets(slug: str) -> dict[str, str]:
    return P2_SNIPPETS[slug]
