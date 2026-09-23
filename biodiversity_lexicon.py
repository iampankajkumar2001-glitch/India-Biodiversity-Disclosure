"""
biodiversity_lexicon.py
------------------------
Ports the "Biodiversity Dictionary" and sentence-selection rules from:

    Giglio, S., Kuchler, T., Stroebel, J. & Zeng, X. (2023),
    "Biodiversity Risk", NBER Working Paper 31137, Section 1.2 and
    Appendix A.4.2.

and extends them for scoring Indian corporate annual / sustainability /
BRSR (Business Responsibility & Sustainability Report) disclosures instead
of US 10-K filings.

Design notes
------------
The paper's dictionary contains some *safe* (unambiguous) unigrams that,
on their own, reliably signal a biodiversity-related sentence (e.g.
"deforestation", "wildlife"), and some *ambiguous* unigrams that need a
companion term in the same sentence to avoid false positives such as
"software ecosystem" or "marine cargo insurance" (see Appendix A.4.2 of
the paper for the exact examples). We reproduce both lists faithfully and
add an optional, togglable set of India-specific ecological terms
(mangroves, Western Ghats, Ramsar wetlands, tiger reserves, etc.) plus an
Indian regulation dictionary (Wildlife Protection Act, Biological
Diversity Act, NGT, MoEFCC, CRZ, etc.) that mirrors the role the paper's
"law/regulation/Act/ESA/discharge/restriction" list plays for the
10K-Biodiversity-Regulation Score.

Everything here is plain Python / regex — no ML dependency — so it can be
unit tested (see tests/test_core.py) without downloading any model
weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. Base "safe" biodiversity terms (Giglio et al. 2023, Section 1.2)
#    These count as a biodiversity match on their own, no companion needed.
#    Patterns are already word-bounded by _compile().
# ---------------------------------------------------------------------------
SAFE_TERM_PATTERNS: list[str] = [
    r"biodiversity",
    r"ecolog\w*",          # ecology, ecological, ecologically
    r"habitats?",
    r"rainforests?",
    r"forests?",
    r"deforestation",
    r"fauna",
    r"flora",
    r"freshwater",
    r"wetlands?",
    r"wildlife",
    r"corals?",
    r"aquatic",
    r"desertification",
    r"carbon\s+sinks?",
    r"ecosphere",
    r"biospher\w*",
]

# ---------------------------------------------------------------------------
# 2. Ambiguous unigrams that need a companion term in the SAME sentence
#    (paper Appendix A.4.2). "ecosystem" and "species" need a companion
#    word anywhere in the sentence; "marine" and "tropical" are only kept
#    when they appear in one of a short list of safe bigrams.
# ---------------------------------------------------------------------------

ECOSYSTEM_PATTERN = r"ecosystems?"
ECOSYSTEM_COMPANIONS = [
    r"climate", r"coast\w*", r"forest\w*", r"micro\w*", r"natur\w*",
    r"public\s+health", r"sustain\w*", r"water",
]

SPECIES_PATTERN = r"species"
SPECIES_COMPANIONS = [
    r"aquatic", r"biodiversity", r"birds?", r"endanger\w*", r"environment\w*",
    r"fish", r"habitats?", r"invasive", r"lists?", r"marine", r"protect\w*",
    r"threat\w*", r"\bESA\b", r"\bEPA\b",
]
# Indian-context companions added to the species disambiguation set when
# india_terms=True — these are the local analogues of "ESA"/"EPA".
SPECIES_COMPANIONS_INDIA = [
    r"wildlife\s+protection\s+act", r"\bWPA\b", r"biological\s+diversity\s+act",
    r"\bNBA\b", r"\biucn\b", r"forest\s+conservation\s+act", r"\bMoEFCC\b",
    r"schedule\s+[iv]+\s+species",
]

# "marine" / "tropical" are only counted through these safe bigrams, exactly
# as in the paper's appendix (this avoids "marine cargo insurance",
# "tropical fruit", etc.)
MARINE_SAFE_PHRASES = [
    r"marine\s+biodiversity", r"marine\s+ecosystems?", r"marine\s+environment",
    r"marine\s+life", r"marine\s+species",
]
TROPICAL_SAFE_PHRASES = [
    r"tropical\s+biodiversity", r"tropical\s+ecosystems?", r"tropical\s+environment",
    r"tropical\s+forests?", r"tropical\s+species",
]

# ---------------------------------------------------------------------------
# 3. Optional India-specific *safe* ecological vocabulary. These are treated
#    like the base safe terms (no companion needed) because they are
#    unambiguous in a corporate-disclosure context. Toggle with
#    india_terms=True.
# ---------------------------------------------------------------------------
INDIA_SAFE_TERM_PATTERNS: list[str] = [
    r"mangroves?",
    r"western\s+ghats?",
    r"eastern\s+ghats?",
    r"sundarbans?",
    r"biodiversity\s+hotspots?",
    r"sacred\s+groves?",
    r"ramsar(\s+site|\s+wetland)?s?",
    r"tiger\s+reserves?",
    r"elephant\s+corridors?",
    r"wildlife\s+sanctuar(y|ies)",
    r"afforestation",
    r"agroforestry",
    r"greenbelt\w*",
    r"catchment\s+area\w*",
    r"riverine",
    r"estuar\w+",
    r"biosphere\s+reserves?",
    r"project\s+tiger",
    r"project\s+elephant",
    r"coastal\s+regulation\s+zone",
    r"\bCRZ\b",
    r"rewilding",
    r"watersheds?",
    r"invasive\s+species",
    r"native\s+species",
    r"endemic\s+species",
]

# ---------------------------------------------------------------------------
# 4. Regulation vocabulary — drives the "Regulation Score"
#    (paper's 10K-Biodiversity-Regulation Score, Section 2.1).
#    Base terms are the paper's law/regulation/Act/ESA/discharge/restriction
#    list; the rest are Indian statutory / regulatory analogues.
# ---------------------------------------------------------------------------
REGULATION_TERM_PATTERNS: list[str] = [
    # base terms used in the paper
    r"laws?", r"regulations?", r"\bacts?\b", r"\bESA\b", r"discharges?",
    r"restrictions?",
    # Indian statutory / regulatory extensions
    r"rules?", r"notifications?", r"tribunals?", r"\bNGT\b",
    r"national\s+green\s+tribunal", r"bans?", r"banned", r"licen[cs]es?",
    r"permits?", r"compliance", r"statut\w*", r"\bMoEFCC\b",
    r"ministry\s+of\s+environment", r"wildlife\s+protection\s+act",
    r"forest\s+conservation\s+act", r"forest\s+\(conservation\)\s+act",
    r"environment\s+\(protection\)\s+act", r"environment\s+protection\s+act",
    r"biological\s+diversity\s+act", r"biodiversity\s+act", r"\bCRZ\b",
    r"coastal\s+regulation\s+zone", r"\bEIA\b",
    r"environmental\s+impact\s+assessment", r"\bCITES\b", r"courts?",
    r"litigation", r"penalt(y|ies)", r"fines?", r"prohibit\w*",
    r"mandat\w*", r"consent\s+to\s+(operate|establish)",
    r"pollution\s+control\s+board", r"\bNBA\b",
    r"national\s+biodiversity\s+authority", r"state\s+biodiversity\s+board",
    r"sanctuary\s+notification",
]


@dataclass
class Lexicon:
    """Compiled regex lexicon used to flag biodiversity / regulation
    sentences. Build with `build_lexicon()`."""

    safe_terms: list[re.Pattern] = field(default_factory=list)
    ecosystem_term: re.Pattern | None = None
    ecosystem_companions: list[re.Pattern] = field(default_factory=list)
    species_term: re.Pattern | None = None
    species_companions: list[re.Pattern] = field(default_factory=list)
    marine_safe_phrases: list[re.Pattern] = field(default_factory=list)
    tropical_safe_phrases: list[re.Pattern] = field(default_factory=list)
    regulation_terms: list[re.Pattern] = field(default_factory=list)
    india_terms_enabled: bool = True


def _compile(pattern: str) -> re.Pattern:
    """Word/phrase-bounded, case-insensitive compile."""
    return re.compile(rf"(?<![A-Za-z]){pattern}(?![A-Za-z])", re.IGNORECASE)


def build_lexicon(
    india_terms: bool = True,
    extra_safe_terms: Iterable[str] | None = None,
    extra_regulation_terms: Iterable[str] | None = None,
) -> Lexicon:
    """Build the compiled lexicon.

    Parameters
    ----------
    india_terms:
        If True, add the India-specific safe ecological vocabulary and the
        Indian statutory companions/regulation terms on top of the paper's
        original dictionary. If False, this reduces to (close to) the
        original Giglio et al. (2023) dictionary.
    extra_safe_terms / extra_regulation_terms:
        Optional user-supplied plain-text terms (e.g. typed into the
        Streamlit sidebar) to append on top of everything else. Multi-word
        phrases are fine ("cauvery river basin"); regex metacharacters are
        escaped automatically.
    """
    safe_patterns = list(SAFE_TERM_PATTERNS)
    species_companions = list(SPECIES_COMPANIONS)
    regulation_patterns = list(REGULATION_TERM_PATTERNS)

    if india_terms:
        safe_patterns += INDIA_SAFE_TERM_PATTERNS
        species_companions += SPECIES_COMPANIONS_INDIA

    if extra_safe_terms:
        safe_patterns += [re.escape(t.strip()) for t in extra_safe_terms if t.strip()]
    if extra_regulation_terms:
        regulation_patterns += [
            re.escape(t.strip()) for t in extra_regulation_terms if t.strip()
        ]

    return Lexicon(
        safe_terms=[_compile(p) for p in safe_patterns],
        ecosystem_term=_compile(ECOSYSTEM_PATTERN),
        ecosystem_companions=[_compile(p) for p in ECOSYSTEM_COMPANIONS],
        species_term=_compile(SPECIES_PATTERN),
        species_companions=[_compile(p) for p in species_companions],
        marine_safe_phrases=[_compile(p) for p in MARINE_SAFE_PHRASES],
        tropical_safe_phrases=[_compile(p) for p in TROPICAL_SAFE_PHRASES],
        regulation_terms=[_compile(p) for p in regulation_patterns],
        india_terms_enabled=india_terms,
    )


def find_biodiversity_matches(sentence: str, lexicon: Lexicon) -> list[str]:
    """Return the list of matched biodiversity terms/phrases in `sentence`
    (empty list if the sentence is not biodiversity-related). Mirrors the
    sentence-selection logic of Giglio et al. (2023) Appendix A.4.2:
    safe terms always count; "ecosystem"/"species" need a companion word
    anywhere in the sentence; "marine"/"tropical" only count through a
    short list of safe bigrams.
    """
    matches: list[str] = []

    for pat in lexicon.safe_terms:
        m = pat.search(sentence)
        if m:
            matches.append(m.group(0))

    if lexicon.ecosystem_term and lexicon.ecosystem_term.search(sentence):
        if any(c.search(sentence) for c in lexicon.ecosystem_companions):
            matches.append(lexicon.ecosystem_term.search(sentence).group(0))

    if lexicon.species_term and lexicon.species_term.search(sentence):
        if any(c.search(sentence) for c in lexicon.species_companions):
            matches.append(lexicon.species_term.search(sentence).group(0))

    for pat in lexicon.marine_safe_phrases:
        m = pat.search(sentence)
        if m:
            matches.append(m.group(0))

    for pat in lexicon.tropical_safe_phrases:
        m = pat.search(sentence)
        if m:
            matches.append(m.group(0))

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def find_regulation_matches(sentence: str, lexicon: Lexicon) -> list[str]:
    """Return matched regulation terms/phrases in `sentence` (paper's
    10K-Biodiversity-Regulation Score logic, Section 2.1)."""
    matches = []
    for pat in lexicon.regulation_terms:
        m = pat.search(sentence)
        if m:
            matches.append(m.group(0))
    seen = set()
    deduped = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped
