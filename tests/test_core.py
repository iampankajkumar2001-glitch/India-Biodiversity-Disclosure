"""
Unit tests for the (pure-Python, no-ML) lexicon and scoring logic.

Run with:  pytest -q
These do NOT require internet access or the transformers/torch/streamlit
dependencies — only the standard library — so they're a fast way to sanity
check any changes you make to the dictionary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biodiversity_lexicon import (
    build_lexicon,
    find_biodiversity_matches,
    find_regulation_matches,
)
from scoring import SentenceRecord, compute_document_scores


# ---------------------------------------------------------------------------
# Lexicon: safe terms
# ---------------------------------------------------------------------------

def test_safe_term_deforestation_matches():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "Deforestation in the upstream catchment has reduced timber supply.", lex
    )
    assert matches, "deforestation should be flagged as a safe term"


def test_safe_term_wildlife_matches():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "The company follows guidelines to protect wildlife near its plants.", lex
    )
    assert matches


# ---------------------------------------------------------------------------
# Lexicon: disambiguation (the whole point of Appendix A.4.2)
# ---------------------------------------------------------------------------

def test_software_ecosystem_is_excluded():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "Our products compete on price, features, and software ecosystem support.",
        lex,
    )
    assert matches == [], "software ecosystem must NOT be flagged"


def test_natural_ecosystem_is_included():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "Mining activity can disturb the natural ecosystem around the site.", lex
    )
    assert matches, "ecosystem + 'natural' companion should be flagged"


def test_marine_cargo_insurance_is_excluded():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "The Company maintains marine cargo insurance covering shipment losses.",
        lex,
    )
    assert matches == [], "marine cargo insurance must NOT be flagged"


def test_marine_ecosystem_is_included():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "Dredging activity may affect the marine ecosystem near the port.", lex
    )
    assert matches


def test_tropical_fruit_is_excluded():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "All of our tropical fruit shipments are delivered using pallets.", lex
    )
    assert matches == []


def test_wood_species_is_included_via_habitat_companion():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "Harvesting is restricted for species that depend on this habitat.", lex
    )
    assert matches


def test_bare_species_without_companion_is_excluded():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "We offer three species of premium coffee blends to our customers.", lex
    )
    assert matches == []


# ---------------------------------------------------------------------------
# Lexicon: India-specific extensions
# ---------------------------------------------------------------------------

def test_india_terms_disabled_by_default_flag():
    lex = build_lexicon(india_terms=False)
    matches = find_biodiversity_matches(
        "The plant is located near the Sundarbans mangroves.", lex
    )
    assert matches == [], "India terms should be off when india_terms=False"


def test_india_terms_enabled_catch_mangroves():
    lex = build_lexicon(india_terms=True)
    matches = find_biodiversity_matches(
        "The plant is located near the Sundarbans mangroves.", lex
    )
    assert matches


def test_wildlife_protection_act_flags_regulation():
    lex = build_lexicon(india_terms=True)
    sentence = (
        "Operations require clearance under the Wildlife Protection Act "
        "and habitat impact assessment."
    )
    biodiv = find_biodiversity_matches(sentence, lex)
    reg = find_regulation_matches(sentence, lex)
    assert biodiv and reg


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_count_score_threshold():
    recs = [
        SentenceRecord(text="a", is_biodiversity=True, sentiment_value=1),
        SentenceRecord(text="b", is_biodiversity=False),
        SentenceRecord(text="c", is_biodiversity=False),
    ]
    scores = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores.count_score == 0  # only 1 biodiversity sentence, below threshold=2

    recs.append(SentenceRecord(text="d", is_biodiversity=True, sentiment_value=-1))
    scores2 = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores2.count_score == 1


def test_negative_score_is_neg_minus_pos():
    recs = [
        SentenceRecord(text="a", is_biodiversity=True, sentiment_value=-1),
        SentenceRecord(text="b", is_biodiversity=True, sentiment_value=-1),
        SentenceRecord(text="c", is_biodiversity=True, sentiment_value=1),
        SentenceRecord(text="d", is_biodiversity=True, sentiment_value=0),
    ]
    scores = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores.negative_score == 1  # 2 negative - 1 positive


def test_no_mention_gives_zero_scores():
    recs = [SentenceRecord(text="a", is_biodiversity=False) for _ in range(5)]
    scores = compute_document_scores("Test Co", "2023", recs)
    assert scores.count_score == 0
    assert scores.negative_score == 0
    assert scores.regulation_score == 0
    assert scores.biodiversity_intensity_pct == 0.0


def test_regulation_score_requires_count_score():
    # A single regulation-flagged biodiversity sentence, below the count
    # threshold, must NOT trigger the regulation score (paper Sec 2.1:
    # regulation score requires >=2 biodiversity sentences).
    recs = [
        SentenceRecord(
            text="a", is_biodiversity=True, is_regulation=True, sentiment_value=-1
        ),
        SentenceRecord(text="b", is_biodiversity=False),
    ]
    scores = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores.count_score == 0
    assert scores.regulation_score == 0

    recs.append(SentenceRecord(text="c", is_biodiversity=True, sentiment_value=0))
    scores2 = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores2.count_score == 1
    assert scores2.regulation_score == 1


def test_biodiversity_intensity_percentage():
    recs = [SentenceRecord(text=str(i), is_biodiversity=(i < 3)) for i in range(10)]
    scores = compute_document_scores("Test Co", "2023", recs, count_threshold=2)
    assert scores.n_biodiversity_sentences == 3
    assert abs(scores.biodiversity_intensity_pct - 30.0) < 1e-9
