"""Tests for putting lost style tags back without calling the LLM."""

from docagent.translate.tags import prune_tags, restore_tags

TERMS = {
    "likely": "probable",
    "very likely": "très probable",
    "high confidence": "degré de confiance élevé",
}


def test_footnote_call_glued_to_a_word():
    source = "Global surface temperature<s1>8</s1> in the first two decades"
    translation = "La température à la surface du globe8 au cours des deux premières décennies"
    assert restore_tags(source, translation, {}) == (
        "La température à la surface du globe<s1>8</s1> au cours des deux premières décennies"
    )


def test_footnote_call_after_a_year_is_found_with_its_anchor():
    # "2010–2019" + "11" became "2010–201911": the year locates the call.
    source = "from 1850–1900 to 2010–2019<s1>11</s1> is 0.8°C"
    translation = "de 1850–1900 à 2010–201911 est de 0,8 °C"
    expected = "de 1850–1900 à 2010–2019<s1>11</s1> est de 0,8 °C"
    assert restore_tags(source, translation, {}) == expected


def test_italic_calibrated_language_through_the_glossary():
    source = "It is <s1>likely</s1> that it warmed (<s1>high confidence</s1>)."
    translation = "Il est probable qu'il se soit réchauffé (degré de confiance élevé)."
    assert restore_tags(source, translation, TERMS) == (
        "Il est <s1>probable</s1> qu'il se soit réchauffé (<s1>degré de confiance élevé</s1>)."
    )


def test_inflected_translation_is_still_found():
    source = "Human influence is <s1>very likely</s1> the main driver"
    translation = "L'influence humaine est très probablement le principal facteur"
    assert restore_tags(source, translation, TERMS) == (
        "L'influence humaine est <s1>très probablement</s1> le principal facteur"
    )


def test_bold_lead_in_ends_at_the_same_punctuation():
    source = "<s1>Panel (a) Observed global warming</s1> (increase in temperature)."
    translation = "Panneau (a) Réchauffement planétaire observé (hausse de la température)."
    assert restore_tags(source, translation, {}) == (
        "<s1>Panneau (a) Réchauffement planétaire observé</s1> (hausse de la température)."
    )


def test_nothing_is_restored_when_a_run_is_missing():
    # The footnote call was dropped by the model: better no style than a wrong one.
    source = "It is <s1>likely</s1> warmer<s2>8</s2> now"
    translation = "Il fait probablement plus chaud maintenant"
    assert restore_tags(source, translation, TERMS) is None


def test_tags_added_by_the_model_are_removed():
    # The model styled a glossary term that is plain in the source.
    source = "It is <s1>likely</s1> the main driver"
    translation = "C'est <s1>probablement</s1> le <s1>principal facteur</s1>"
    assert prune_tags(source, translation, TERMS) == (
        "C'est <s1>probablement</s1> le principal facteur"
    )


def test_untagged_source_gets_an_untagged_translation():
    source = "Throughout this SPM, main driver means more than 50%"
    translation = "Dans ce RID, « <s1>principal facteur</s1> » signifie plus de 50 %"
    assert prune_tags(source, translation, {}) == (
        "Dans ce RID, « principal facteur » signifie plus de 50 %"
    )


def test_pruning_gives_up_when_a_source_run_is_missing():
    source = "Temperature<s1>8</s1> rose"
    translation = "La <s1>température</s1> a augmenté"
    assert prune_tags(source, translation, {}) is None
