"""
app.py — India BioRisk
=======================
Firm-level biodiversity risk exposure, measured from Indian annual /
sustainability / BRSR reports.

Extends the methodology of:
    Giglio, S., Kuchler, T., Stroebel, J. & Zeng, X. (2023),
    "Biodiversity Risk", NBER Working Paper 31137.
from US 10-K filings to Indian corporate disclosures, using a BERT-based
sentence-sentiment classifier exactly as in the paper's Section 1.2.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from biodiversity_lexicon import (
    build_lexicon,
    find_biodiversity_matches,
    find_regulation_matches,
)
from nlp_utils import (
    SENTIMENT_MODELS,
    classify_sentences_sentiment,
    clean_extracted_text,
    extract_text_from_pdf,
    load_sentiment_pipeline,
    split_into_sentences,
)
from scoring import DocumentScores, SentenceRecord, compute_document_scores
from sample_data.sample_reports import SAMPLE_REPORTS

st.set_page_config(
    page_title="India BioRisk — Firm-Level Biodiversity Risk Exposure",
    page_icon="🌿",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached, expensive resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline(model_key: str):
    return load_sentiment_pipeline(model_key)


@st.cache_data(show_spinner=False)
def cached_extract_pdf_text(file_bytes: bytes) -> str:
    return extract_text_from_pdf(file_bytes)


def guess_company_year(filename: str) -> tuple[str, str]:
    """Best-effort parse of 'Company_2023.pdf' / 'Company-2023.pdf' style
    filenames into (company, year); falls back to sensible defaults."""
    stem = filename.rsplit(".", 1)[0]
    for sep in ["_", "-"]:
        if sep in stem:
            parts = stem.split(sep)
            if parts[-1].strip().isdigit() and len(parts[-1].strip()) == 4:
                return sep.join(parts[:-1]).replace("_", " ").strip(), parts[-1].strip()
    return stem.replace("_", " ").strip(), str(datetime.now().year)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

st.sidebar.title("🌿 India BioRisk")
st.sidebar.caption(
    "Firm-level biodiversity exposure from Indian annual / sustainability / "
    "BRSR reports, extending Giglio, Kuchler, Stroebel & Zeng (2023)."
)

st.sidebar.header("1. Load reports")
input_mode = st.sidebar.radio(
    "Input source",
    ["Upload PDF(s)", "Paste text", "Try sample data"],
    index=2,
)

docs_to_process: list[dict] = []  # each: {company, year, text}

if input_mode == "Upload PDF(s)":
    uploaded_files = st.sidebar.file_uploader(
        "Annual / Sustainability / BRSR report (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.sidebar.caption("Confirm / edit the Company and Year for each file:")
        for f in uploaded_files:
            default_company, default_year = guess_company_year(f.name)
            c1, c2 = st.sidebar.columns(2)
            company = c1.text_input(
                "Company", value=default_company, key=f"company_{f.name}"
            )
            year = c2.text_input("Year", value=default_year, key=f"year_{f.name}")
            docs_to_process.append({"company": company, "year": year, "file": f})

elif input_mode == "Paste text":
    pasted_company = st.sidebar.text_input("Company", value="My Company")
    pasted_year = st.sidebar.text_input("Year", value=str(datetime.now().year))
    pasted_text = st.sidebar.text_area("Paste report text", height=200)
    if pasted_text.strip():
        docs_to_process.append(
            {"company": pasted_company, "year": pasted_year, "text": pasted_text}
        )

else:  # Try sample data
    chosen = st.sidebar.multiselect(
        "Sample company-year reports (fictional, illustrative only)",
        options=list(SAMPLE_REPORTS.keys()),
        default=list(SAMPLE_REPORTS.keys()),
    )
    for key in chosen:
        company, year = key.split("|")
        docs_to_process.append({"company": company, "year": year, "text": SAMPLE_REPORTS[key]})

st.sidebar.header("2. Dictionary options")
india_terms = st.sidebar.checkbox(
    "Include India-specific ecological & regulatory terms", value=True
)
extra_safe_raw = st.sidebar.text_area(
    "Custom biodiversity terms (comma-separated, optional)", value=""
)
extra_reg_raw = st.sidebar.text_area(
    "Custom regulation terms (comma-separated, optional)", value=""
)
count_threshold = st.sidebar.slider(
    "Minimum biodiversity sentences to flag a report", min_value=1, max_value=10, value=2
)

st.sidebar.header("3. Sentiment model")
model_key = st.sidebar.selectbox(
    "BERT sentiment model",
    options=list(SENTIMENT_MODELS.keys()),
    format_func=lambda k: SENTIMENT_MODELS[k].display_name,
)
st.sidebar.caption(
    "First run downloads model weights (a few hundred MB) and needs internet "
    "access; later runs reuse the cached model."
)

run_clicked = st.sidebar.button("▶ Run analysis", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = {}  # key -> {"scores": DocumentScores, "records": [...]}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_document(company: str, year: str, text: str, lexicon, pipe, model_key: str,
                      threshold: int) -> tuple[DocumentScores, list[SentenceRecord]]:
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
        sentiments = classify_sentences_sentiment(pipe, biodiv_sentences, model_key)
        it = iter(sentiments)
        for r in records:
            if r.is_biodiversity:
                label, value = next(it)
                r.sentiment_label = label
                r.sentiment_value = value

    scores = compute_document_scores(company, year, records, count_threshold=threshold)
    return scores, records


if run_clicked:
    if not docs_to_process:
        st.sidebar.error("Add at least one report before running.")
    else:
        extra_safe = [t for t in extra_safe_raw.split(",") if t.strip()]
        extra_reg = [t for t in extra_reg_raw.split(",") if t.strip()]
        lexicon = build_lexicon(
            india_terms=india_terms,
            extra_safe_terms=extra_safe,
            extra_regulation_terms=extra_reg,
        )

        with st.spinner(f"Loading {SENTIMENT_MODELS[model_key].display_name} ..."):
            pipe = get_pipeline(model_key)

        progress = st.progress(0.0, text="Processing reports...")
        new_results = {}
        for i, doc in enumerate(docs_to_process):
            key = f"{doc['company']}|{doc['year']}"
            if "file" in doc:
                file_bytes = doc["file"].getvalue()
                text = cached_extract_pdf_text(file_bytes)
            else:
                text = doc["text"]

            scores, records = process_document(
                doc["company"], doc["year"], text, lexicon, pipe, model_key, count_threshold
            )
            new_results[key] = {"scores": scores, "records": records}
            progress.progress((i + 1) / len(docs_to_process), text=f"Processed {doc['company']} ({doc['year']})")

        st.session_state["results"] = new_results
        progress.empty()
        st.sidebar.success(f"Processed {len(new_results)} report(s).")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🌿 India BioRisk — Firm-Level Biodiversity Risk Exposure")
st.caption(
    "A research tool that scores biodiversity risk disclosure in Indian annual, "
    "sustainability, and BRSR reports, extending the sentence-level dictionary + "
    "BERT-sentiment approach of Giglio, Kuchler, Stroebel & Zeng (2023), *Biodiversity "
    "Risk*, NBER WP 31137, from US 10-K filings to the Indian context."
)

results = st.session_state["results"]

if not results:
    st.info(
        "Use the sidebar to upload one or more PDF reports (or try the built-in sample "
        "data), then click **Run analysis**."
    )
else:
    summary_rows = [r["scores"].as_dict() for r in results.values()]
    summary_df = pd.DataFrame(summary_rows)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Summary", "📈 Cross-firm charts", "🔍 Sentence explorer", "☁️ Word cloud", "📘 Methodology"]
    )

    # -- Tab 1: Summary ------------------------------------------------
    with tab1:
        st.subheader("Report-level scores")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="biodiversity_risk_summary.csv",
                mime="text/csv",
            )
        with col2:
            all_sentence_rows = []
            for key, r in results.items():
                company, year = key.split("|")
                for rec in r["records"]:
                    if rec.is_biodiversity:
                        all_sentence_rows.append(
                            {
                                "Company": company,
                                "Year": year,
                                "Sentence": rec.text,
                                "Matched terms": ", ".join(rec.matched_terms),
                                "Sentiment": rec.sentiment_label,
                                "Regulation-flagged": rec.is_regulation,
                                "Regulation terms": ", ".join(rec.regulation_terms),
                            }
                        )
            sentence_df = pd.DataFrame(all_sentence_rows)
            st.download_button(
                "⬇ Download sentence-level data (CSV)",
                data=sentence_df.to_csv(index=False).encode("utf-8"),
                file_name="biodiversity_risk_sentences.csv",
                mime="text/csv",
                disabled=sentence_df.empty,
            )

    # -- Tab 2: Charts ---------------------------------------------------
    with tab2:
        st.subheader("Biodiversity intensity by report")
        chart_df = summary_df.set_index(
            summary_df["Company"] + " (" + summary_df["Year"].astype(str) + ")"
        )["Biodiversity intensity (%)"]
        st.bar_chart(chart_df)

        st.subheader("Sentiment composition of biodiversity sentences")
        sentiment_df = summary_df.set_index(
            summary_df["Company"] + " (" + summary_df["Year"].astype(str) + ")"
        )[["Positive sentences", "Negative sentences", "Neutral sentences"]]
        st.bar_chart(sentiment_df)

        # Time series if any company has more than one year
        multi_year_companies = summary_df.groupby("Company").filter(lambda g: len(g) > 1)
        if not multi_year_companies.empty:
            st.subheader("Biodiversity intensity over time (companies with 2+ reports)")
            pivot = multi_year_companies.pivot_table(
                index="Year", columns="Company", values="Biodiversity intensity (%)"
            )
            st.line_chart(pivot)

    # -- Tab 3: Sentence explorer -----------------------------------------
    with tab3:
        st.subheader("Sentence-level explorer")
        selected_key = st.selectbox("Company (Year)", options=list(results.keys()))
        sentiment_filter = st.radio(
            "Filter by sentiment", ["All", "positive", "neutral", "negative"], horizontal=True
        )
        regulation_only = st.checkbox("Show only regulation-flagged sentences")

        records = results[selected_key]["records"]
        biodiv_records = [r for r in records if r.is_biodiversity]
        if sentiment_filter != "All":
            biodiv_records = [r for r in biodiv_records if r.sentiment_label == sentiment_filter]
        if regulation_only:
            biodiv_records = [r for r in biodiv_records if r.is_regulation]

        st.caption(f"{len(biodiv_records)} of {len(records)} total sentences match your filters.")
        badge = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        for r in biodiv_records:
            icon = badge.get(r.sentiment_label, "⚪")
            reg_tag = " 🏛️ *regulation*" if r.is_regulation else ""
            st.markdown(f"{icon} **[{', '.join(r.matched_terms)}]**{reg_tag} — {r.text}")

    # -- Tab 4: Word cloud --------------------------------------------------
    with tab4:
        st.subheader("Biodiversity vocabulary word cloud")
        wc_key = st.selectbox(
            "Company (Year)", options=list(results.keys()), key="wc_select"
        )
        wc_records = [r for r in results[wc_key]["records"] if r.is_biodiversity]
        wc_text = " ".join(r.text for r in wc_records)

        if not wc_text.strip():
            st.info("No biodiversity-related sentences found for this report.")
        else:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                wc = WordCloud(
                    width=1000, height=500, background_color="white",
                    collocations=False,
                ).generate(wc_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except ImportError:
                st.warning(
                    "Install the optional `wordcloud` package to enable this view: "
                    "`pip install wordcloud`."
                )

    # -- Tab 5: Methodology --------------------------------------------------
    with tab5:
        st.markdown(
            """
### How this tool works

This app adapts the methodology of **Giglio, Kuchler, Stroebel & Zeng (2023),
*Biodiversity Risk*, NBER Working Paper 31137**, from US 10-K filings to Indian
corporate annual / sustainability / BRSR reports.

**1. Sentence selection.** Every sentence in a report is checked against a
*Biodiversity Dictionary* (the paper's Appendix A.4.2 list — *biodiversity,
ecosystem(s), ecology/ecological, habitat(s), species, (rain)forest(s),
deforestation, fauna, flora, marine, tropical, freshwater, wetland, wildlife,
coral, aquatic, desertification, carbon sink(s), ecosphere, biosphere* —
plus an optional India-specific list: *mangroves, Western Ghats, Ramsar
wetlands, tiger reserves, elephant corridors, sacred groves,* etc.).
Ambiguous unigrams ("ecosystem", "species") only count when paired with a
companion term in the same sentence, and "marine"/"tropical" only count
through a short list of safe bigrams — this is exactly how the paper
excludes false positives like *"software ecosystem"* or *"marine cargo
insurance"*.

**2. Report-level flag.** A report "covers biodiversity" (**Count Score = 1**)
if it contains at least *N* biodiversity-related sentences (default N = 2,
matching the paper).

**3. Sentiment.** Every biodiversity-related sentence is classified as
positive / neutral / negative using a BERT model — the same BERT
architecture the paper uses (Devlin et al., 2018) — applied here directly
to the report text. The default is **FinBERT**, a BERT-base model
fine-tuned on financial text, since it captures the disclosure-style
language of annual reports better than a generic sentiment model.
**Negative Score** = (# negative sentences) − (# positive sentences),
exactly as in the paper.

**4. Regulation Score.** = 1 if Count Score = 1 **and** at least one
biodiversity sentence also contains a regulation term (the paper's
law/regulation/Act/discharge/restriction list, extended here with Indian
statutes and bodies: the Wildlife Protection Act, the Forest (Conservation)
Act, the Environment (Protection) Act, the Biological Diversity Act, the
National Green Tribunal, MoEFCC, CRZ notifications, State Pollution
Control Boards, etc.).

**5. Biodiversity Intensity** (not in the original paper, which aggregates
across thousands of 10-Ks) = share of a single report's sentences that are
biodiversity-related. This is the more informative continuous measure
this single-report tool uses for its charts.

### What's different from the paper, and why
- **Unit of analysis**: one annual/sustainability/BRSR report at a time
  instead of a corpus of 10-Ks — Indian firms don't file a 10-K
  equivalent, so BRSR/sustainability disclosures (mandatory for the top
  1,000 listed companies by market cap since FY2022-23 under SEBI's BRSR
  framework) are the closest analogue.
- **Dictionary**: extended with India-specific ecological vocabulary and
  Indian environmental statutes, since the paper's "ESA"/"EPA" terms are
  US-specific.
- **Aggregate industry-level exposure and asset-pricing tests** (Sections
  2–3 of the paper, e.g. the hedging-portfolio analysis) require a large
  panel of firms and returns data and are **out of scope for this
  single-document tool** — `batch_score.py` in this project is meant to
  help you build that panel across many firm-years, which you can then
  aggregate to the industry level and merge with return data yourself,
  following the paper's Section 2–3 approach.

### Limitations
- Regex/dictionary matching and off-the-shelf sentiment models are noisy;
  always spot-check flagged sentences (the Sentence Explorer tab) before
  drawing conclusions.
- PDF text extraction from scanned/older reports can be imperfect; an
  optional OCR fallback is included but requires `pytesseract` +
  `pdf2image` + the Tesseract/Poppler system binaries.
- This is a research/screening tool, not investment advice.

**Citation**: Giglio, S., Kuchler, T., Stroebel, J. & Zeng, X. (2023).
*Biodiversity Risk*. NBER Working Paper 31137.
            """
        )
