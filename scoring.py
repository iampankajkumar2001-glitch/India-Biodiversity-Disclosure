"""
scoring.py
----------
Turns sentence-level biodiversity/regulation/sentiment flags into the
report-level scores, mirroring the three 10-K-based measures in
Giglio, Kuchler, Stroebel & Zeng (2023), Section 2.1:

  * 10K-Biodiversity-Count Score      -> Report-Biodiversity-Count Score
  * 10K-Biodiversity-Negative Score   -> Report-Biodiversity-Negative Score
  * 10K-Biodiversity-Regulation Score -> Report-Biodiversity-Regulation Score

plus one continuous measure not in the original paper (which only reports
a binary Count Score because 10-Ks are aggregated across thousands of
firms): Biodiversity Intensity, the share of a single report's sentences
that are biodiversity-related. For a one-report-at-a-time tool this is
more informative than a 0/1 flag and is what the dashboard bar charts
and time series use.

Pure Python / dataclasses — no ML or Streamlit dependency, so it is unit
tested directly (see tests/test_core.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SentenceRecord:
    text: str
    is_biodiversity: bool = False
    matched_terms: list[str] = field(default_factory=list)
    is_regulation: bool = False
    regulation_terms: list[str] = field(default_factory=list)
    sentiment_label: str | None = None   # "positive" | "neutral" | "negative"
    sentiment_value: int = 0             # +1 / 0 / -1, BERT-assigned (paper Sec 1.2)


@dataclass
class DocumentScores:
    company: str
    year: str
    n_sentences: int
    n_biodiversity_sentences: int
    biodiversity_intensity_pct: float
    count_score: int                 # 0/1, threshold-based, paper Sec 2.1
    negative_score: int              # #negative - #positive, paper Sec 2.1
    regulation_score: int            # 0/1, paper Sec 2.1
    n_positive: int
    n_negative: int
    n_neutral: int
    pct_negative_of_biodiversity: float
    pct_positive_of_biodiversity: float
    n_regulation_sentences: int

    def as_dict(self) -> dict:
        return {
            "Company": self.company,
            "Year": self.year,
            "Sentences (total)": self.n_sentences,
            "Biodiversity sentences": self.n_biodiversity_sentences,
            "Biodiversity intensity (%)": round(self.biodiversity_intensity_pct, 3),
            "Count Score": self.count_score,
            "Negative Score": self.negative_score,
            "Regulation Score": self.regulation_score,
            "Positive sentences": self.n_positive,
            "Negative sentences": self.n_negative,
            "Neutral sentences": self.n_neutral,
            "% negative (of biodiv. sentences)": round(self.pct_negative_of_biodiversity, 1),
            "% positive (of biodiv. sentences)": round(self.pct_positive_of_biodiversity, 1),
            "Regulation-flagged sentences": self.n_regulation_sentences,
        }


def compute_document_scores(
    company: str,
    year: str,
    sentence_records: list[SentenceRecord],
    count_threshold: int = 2,
) -> DocumentScores:
    """Aggregate sentence-level records into the report-level scores.

    `count_threshold` mirrors the paper's rule (Section 2.1): a report is
    flagged as covering biodiversity ("Count Score" = 1) only if it
    contains at least `count_threshold` biodiversity-related sentences
    (the paper uses 2, following its treatment of both NYT articles and
    10-K statements).
    """
    n_sentences = len(sentence_records)
    biodiv = [s for s in sentence_records if s.is_biodiversity]
    n_biodiv = len(biodiv)

    count_score = 1 if n_biodiv >= count_threshold else 0

    n_positive = sum(1 for s in biodiv if s.sentiment_value > 0)
    n_negative = sum(1 for s in biodiv if s.sentiment_value < 0)
    n_neutral = n_biodiv - n_positive - n_negative

    # Footnote 8 of the paper: firms with no mention, or only neutral
    # mentions, get a Negative Score of 0. That falls out automatically
    # here since n_positive == n_negative == 0 in both cases.
    negative_score = n_negative - n_positive

    n_regulation_sentences = sum(1 for s in biodiv if s.is_regulation)
    regulation_score = 1 if (count_score == 1 and n_regulation_sentences >= 1) else 0

    intensity = (n_biodiv / n_sentences * 100.0) if n_sentences else 0.0
    pct_neg = (n_negative / n_biodiv * 100.0) if n_biodiv else 0.0
    pct_pos = (n_positive / n_biodiv * 100.0) if n_biodiv else 0.0

    return DocumentScores(
        company=company,
        year=year,
        n_sentences=n_sentences,
        n_biodiversity_sentences=n_biodiv,
        biodiversity_intensity_pct=intensity,
        count_score=count_score,
        negative_score=negative_score,
        regulation_score=regulation_score,
        n_positive=n_positive,
        n_negative=n_negative,
        n_neutral=n_neutral,
        pct_negative_of_biodiversity=pct_neg,
        pct_positive_of_biodiversity=pct_pos,
        n_regulation_sentences=n_regulation_sentences,
    )
