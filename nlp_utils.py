"""
nlp_utils.py
------------
PDF text extraction, sentence segmentation, and the BERT sentiment
classifier used to reproduce the paper's sentence-level positive /
negative / neutral labelling (Giglio et al. 2023, Section 1.2: "we adopt
the Bidirectional Encoder Representations from Transformers (BERT) model
... to classify each of the selected biodiversity sentences").

This module is the only place that touches `pdfplumber` / `transformers`
/ `pysbd`, all of which are optional at import time (guarded) so that
`biodiversity_lexicon.py` and `scoring.py` stay independently testable.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF (file path, bytes, or a file-like object such
    as Streamlit's UploadedFile). Falls back to OCR (if pytesseract +
    pdf2image + poppler are installed) for pages that yield no text — this
    matters for scanned/older annual reports, which are common in India.
    """
    import pdfplumber

    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = io.BytesIO(file_obj)

    pages_text: list[str] = []
    scanned_page_indices: list[int] = []

    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 20:
                scanned_page_indices.append(i)
            pages_text.append(text)

    # Optional OCR fallback for pages that extracted (almost) no text.
    if scanned_page_indices:
        ocr_text_by_page = _try_ocr_fallback(file_obj, scanned_page_indices)
        for idx, ocr_text in ocr_text_by_page.items():
            if ocr_text.strip():
                pages_text[idx] = ocr_text

    return clean_extracted_text("\n".join(pages_text))


def _try_ocr_fallback(file_obj, page_indices: list[int]) -> dict[int, str]:
    """Best-effort OCR for scanned pages. Silently returns {} if
    pytesseract / pdf2image / the poppler & tesseract system binaries are
    not available — OCR is a nice-to-have, not a hard requirement."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
        images = convert_from_bytes(raw)
        out = {}
        for idx in page_indices:
            if idx < len(images):
                out[idx] = pytesseract.image_to_string(images[idx])
        return out
    except Exception:
        return {}


_PARA_MARKER = " \u00b6 "  # pilcrow, used as an internal paragraph-break marker


def clean_extracted_text(text: str) -> str:
    """Undo common PDF-extraction artifacts before sentence splitting:
    de-hyphenate words broken across a line break, mark paragraph breaks
    (handled as hard sentence boundaries by `split_into_sentences`, so we
    never have to guess whether a heading/bullet ended with punctuation),
    collapse repeated whitespace, and drop bare page-number lines."""
    # de-hyphenate: "biodiver-\nsity" -> "biodiversity"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", _PARA_MARKER, text)
    text = re.sub(r"\n", " ", text)
    # drop stand-alone page-number tokens like " 12 " or " Page 12 "
    text = re.sub(r"\bpage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "co", "corp", "no", "fig", "approx", "govt", "dept",
    "u.s", "u.k", "rs", "no", "resp",
}


def split_into_sentences(text: str) -> list[str]:
    """Split cleaned report text into sentences. Uses `pysbd` if it is
    installed (recommended — much better handling of abbreviations,
    decimals, and bullet lists); otherwise falls back to a conservative
    regex splitter that special-cases a short abbreviation list so
    "the Wildlife Protection Act, 1972 (No. 53) ..." isn't split mid
    sentence. Paragraph breaks inserted by `clean_extracted_text` are
    always treated as hard sentence boundaries first, so a section
    heading glued to body text can't silently swallow a neighbouring
    sentence."""
    paragraphs = [p.strip() for p in text.split(_PARA_MARKER.strip()) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    all_sentences: list[str] = []
    for para in paragraphs:
        try:
            import pysbd

            seg = pysbd.Segmenter(language="en", clean=True)
            all_sentences.extend(seg.segment(para))
        except Exception:
            all_sentences.extend(_regex_split(para))

    # drop near-empty / boilerplate fragments
    return [s.strip() for s in all_sentences if len(s.strip()) > 3]


def _regex_split(text: str) -> list[str]:
    # Protect abbreviation periods from being treated as sentence
    # boundaries, without disturbing the original capitalisation (which
    # the boundary-detection lookahead below relies on).
    protected = text
    for abbr in _ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbr)}\.(?=\s)", re.IGNORECASE)
        protected = pattern.sub(lambda m: m.group(0)[:-1] + "<DOT>", protected)
    # Protect decimal numbers ("3.8%") from being split.
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", protected)
    return [s.replace("<DOT>", ".") for s in raw_sentences]


# ---------------------------------------------------------------------------
# BERT sentiment classification
# ---------------------------------------------------------------------------

@dataclass
class SentimentModelSpec:
    display_name: str
    hf_model_id: str
    label_mapper: Callable[[str], str]   # raw model label -> "positive"/"neutral"/"negative"


def _finbert_label_mapper(raw_label: str) -> str:
    # ProsusAI/finbert already outputs {positive, negative, neutral}
    return raw_label.lower()


def _nlptown_label_mapper(raw_label: str) -> str:
    # nlptown/bert-base-multilingual-uncased-sentiment outputs "1 star".."5 stars"
    stars = int(raw_label[0])
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


SENTIMENT_MODELS: dict[str, SentimentModelSpec] = {
    "finbert": SentimentModelSpec(
        display_name="FinBERT (BERT fine-tuned on financial text — recommended)",
        hf_model_id="ProsusAI/finbert",
        label_mapper=_finbert_label_mapper,
    ),
    "multilingual": SentimentModelSpec(
        display_name="Multilingual BERT (nlptown, 5-star bucketed — use for mixed Hindi/English reports)",
        hf_model_id="nlptown/bert-base-multilingual-uncased-sentiment",
        label_mapper=_nlptown_label_mapper,
    ),
}

_LABEL_TO_VALUE = {"positive": 1, "neutral": 0, "negative": -1}


def load_sentiment_pipeline(model_key: str):
    """Load the HF `text-classification` pipeline for the chosen model.
    Wrap this call in `st.cache_resource` in the Streamlit app so the
    (large) model weights are downloaded/loaded only once per session."""
    from transformers import pipeline

    spec = SENTIMENT_MODELS[model_key]
    return pipeline(
        "text-classification",
        model=spec.hf_model_id,
        tokenizer=spec.hf_model_id,
        truncation=True,
        max_length=512,
    )


def classify_sentences_sentiment(
    pipe, sentences: list[str], model_key: str, batch_size: int = 16
) -> list[tuple[str, int]]:
    """Classify each sentence and return (label, value) pairs where label
    is one of "positive"/"neutral"/"negative" and value is +1/0/-1 —
    exactly the scoring convention in Giglio et al. (2023), Section 1.2."""
    if not sentences:
        return []

    spec = SENTIMENT_MODELS[model_key]
    results: list[tuple[str, int]] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        raw_outputs = pipe(batch)
        for out in raw_outputs:
            label = spec.label_mapper(out["label"])
            results.append((label, _LABEL_TO_VALUE.get(label, 0)))
    return results
