"""Tests for spaCy linguistic heuristics."""

from cas_rag.linguistic.spacy_analyzer import SpacyLinguisticAnalyzer


def test_resolved_demonstrative_does_not_force_rag():
    analyzer = SpacyLinguisticAnalyzer()
    comment = "Those kids were so sweet and brave to help that struggling student."
    assert analyzer.needs_rag(comment, "tutoring diner experiment") is False


def test_unresolved_pronoun_still_needs_rag():
    analyzer = SpacyLinguisticAnalyzer()
    comment = "When she walked over I knew exactly what she meant."
    assert analyzer.needs_rag(comment, "tutoring diner experiment") is True


def test_cd_policy_video_mismatch_is_infeasible():
    analyzer = SpacyLinguisticAnalyzer()
    policy = analyzer.cd_rag_policy(
        "The harmonica guy made me cry",
        "A math tutoring session in a diner.",
        "The tutor is harsh. Harper helps with addition.",
    )
    assert policy == "infeasible"


def test_cd_policy_grounded_scene_quote_can_retrieve():
    analyzer = SpacyLinguisticAnalyzer()
    script = "A bystander tells the victim you keep reading while bullies mock her."
    scenario = "Public bus stop bullying experiment."
    policy = analyzer.cd_rag_policy("When she said you keep reading I melted", scenario, script)
    assert policy in ("resolvable", "unnecessary")
