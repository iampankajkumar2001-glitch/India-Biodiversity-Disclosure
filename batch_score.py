#!/usr/bin/env python3
"""
batch_score.py
--------------
Command-line batch scorer for building a firm-year panel dataset from a
folder of PDF annual/sustainability/BRSR reports — the research-panel
counterpart to the interactive `app.py` dashboard, closer to how Giglio
et al. (2023) built their 10K-based panel across many firm-years
(Section 2.2 / Figure 4 of the paper).

Expected input: a folder of PDFs named "Company_Year.pdf" or
"Company-Year.pdf" (e.g. "TataSteel_2022.pdf"). Anything that doesn't
match that pattern is scored with Year="unknown".

Usage
-----
    python batch_score.py --input-dir ./reports --output panel.csv
    python batch_score.py --input-dir ./reports --output panel.csv \\
        --sentences-output sentences.csv --no-india-terms --threshold 3 \\
        --model finbert

Output
------
A CSV with one row per report (Company, Year, and all scores from
`scoring.DocumentScores`), suitable for merging into an industry-level
panel the way the paper aggregates firm-level 10-K scores to industries
(Section 2.3) via value-weighted averaging.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from biodiversity_lexicon import build_lexicon, find_biodiversity_matches, find_regulation_matches
from nlp_utils import (
    SENTIMENT_MODELS,
    classify_sentences_sentiment,
    clean_extracted_text,
    extract_text_from_pdf,
    load_sentiment_pipeline,
    split_into_sentences,
)
from scoring import SentenceRecord, compute_document_scores


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    m = re.match(r"^(.*?)[_-](\d{4})$", stem)
    if m:
        return m.group(1).replace("_", " ").strip(), m.group(2)
    return stem.replace("_", " ").strip(), "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder of PDF reports")
    parser.add_argument("--output", required=True, help="Output CSV path for the firm-year panel")
    parser.add_argument("--sentences-output", default=None, help="Optional CSV path for full sentence-level output")
    parser.add_argument("--no-india-terms", action="store_true", help="Disable India-specific dictionary extensions")
    parser.add_argument("--threshold", type=int, default=2, help="Min. biodiversity sentences to flag a report (default: 2)")
    parser.add_argument("--model", default="finbert", choices=list(SENTIMENT_MODELS.keys()), help="Sentiment model key")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {input_dir}", file=sys.stderr)
        return 1

    lexicon = build_lexicon(india_terms=not args.no_india_terms)
    print(f"Loading sentiment model: {SENTIMENT_MODELS[args.model].display_name} ...")
    pipe = load_sentiment_pipeline(args.model)

    panel_rows = []
    sentence_rows = []

    for i, pdf_path in enumerate(pdf_paths, start=1):
        company, year = parse_filename(pdf_path)
        print(f"[{i}/{len(pdf_paths)}] {pdf_path.name} -> company='{company}', year='{year}'")

        text = extract_text_from_pdf(str(pdf_path))
        cleaned = clean_extracted_text(text)
        sentences = split_into_sentences(cleaned)

        records: list[SentenceRecord] = []
        for s in sentences:
            biodiv_matches = find_biodiversity_matches(s, lexicon)
            rec = SentenceRecord(text=s, is_biodiversity=bool(biodiv_matches), matched_terms=biodiv_matches)
            if biodiv_matches:
                reg_matches = find_regulation_matches(s, lexicon)
                rec.is_regulation = bool(reg_matches)
                rec.regulation_terms = reg_matches
            records.append(rec)

        biodiv_sentences = [r.text for r in records if r.is_biodiversity]
        if biodiv_sentences:
            sentiments = classify_sentences_sentiment(pipe, biodiv_sentences, args.model)
            it = iter(sentiments)
            for r in records:
                if r.is_biodiversity:
                    label, value = next(it)
                    r.sentiment_label = label
                    r.sentiment_value = value

        scores = compute_document_scores(company, year, records, count_threshold=args.threshold)
        panel_rows.append(scores.as_dict())

        if args.sentences_output:
            for r in records:
                if r.is_biodiversity:
                    sentence_rows.append(
                        {
                            "Company": company,
                            "Year": year,
                            "Sentence": r.text,
                            "Matched terms": ", ".join(r.matched_terms),
                            "Sentiment": r.sentiment_label,
                            "Regulation-flagged": r.is_regulation,
                            "Regulation terms": ", ".join(r.regulation_terms),
                        }
                    )

    pd.DataFrame(panel_rows).to_csv(args.output, index=False)
    print(f"\nWrote firm-year panel ({len(panel_rows)} rows) to {args.output}")

    if args.sentences_output:
        pd.DataFrame(sentence_rows).to_csv(args.sentences_output, index=False)
        print(f"Wrote sentence-level data ({len(sentence_rows)} rows) to {args.sentences_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
