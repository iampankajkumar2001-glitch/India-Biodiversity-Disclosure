# India BioRisk 🌿
Firm-level biodiversity risk exposure from Indian annual / sustainability / BRSR reports.

This project extends the methodology of **Giglio, Kuchler, Stroebel & Zeng (2023),
*Biodiversity Risk*, NBER Working Paper 31137** — which measures firm-level
biodiversity risk exposure from **US 10-K filings** using a biodiversity
dictionary + BERT sentiment classifier — to **Indian corporate disclosures**
(annual reports, standalone sustainability reports, and SEBI-mandated BRSR
reports).

> This is a research/screening tool, not investment advice. Dictionary
> matching and off-the-shelf sentiment models are noisy — always spot-check
> flagged sentences in the app's Sentence Explorer tab before drawing
> conclusions.

---

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Interactive Streamlit dashboard (upload PDFs, run the pipeline, explore results) |
| `biodiversity_lexicon.py` | The biodiversity dictionary, disambiguation rules, and regulation-term dictionary (pure Python/regex, no ML) |
| `nlp_utils.py` | PDF text extraction, sentence segmentation, and the BERT sentiment-model wrapper |
| `scoring.py` | Turns sentence-level flags into the report-level scores (pure Python, no ML) |
| `batch_score.py` | CLI tool to build a firm-year panel CSV from a folder of PDFs (for larger research panels) |
| `sample_data/sample_reports.py` | Three fictional example report excerpts used by the app's "Try sample data" mode |
| `tests/test_core.py` | Unit tests for the lexicon and scoring logic — no internet or ML dependencies needed |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# sanity-check the dictionary & scoring logic (fast, no internet needed)
pytest -q

# launch the app
streamlit run app.py
```

The first time you run an analysis, `transformers` will download the chosen
BERT sentiment model (FinBERT is ~400MB) — this needs internet access once;
after that it's cached locally.

### Try it without any files
Pick **"Try sample data"** in the sidebar and click **Run analysis** — this
scores three short, fictional example report excerpts (a beverage company,
a mining company, and a renewable-energy company) so you can see the
dictionary matching, sentiment classification, and all three scores working
end to end before uploading real reports.

### Scoring your own reports
1. Choose **"Upload PDF(s)"** in the sidebar and add one or more annual /
   sustainability / BRSR report PDFs.
2. If a filename follows `Company_2023.pdf` / `Company-2023.pdf`, the
   Company and Year fields auto-fill; otherwise edit them directly.
3. Click **Run analysis**.
4. Use the tabs to review the **Summary** table and scores, **Cross-firm
   charts**, the **Sentence explorer** (read exactly which sentences drove
   the score, filter by sentiment or regulation flag), and a **Word cloud**
   of each report's biodiversity vocabulary (the India analogue of the
   paper's Appendix Figure A.3 sector word clouds).

### Building a larger research panel
For a panel across many firms/years (closer to the paper's Section 2.2/2.3
analysis), name your PDFs `Company_Year.pdf` and run:

```bash
python batch_score.py --input-dir ./reports --output panel.csv \
    --sentences-output sentences.csv
```

This produces one row per report with all the scores below, ready to merge
with a Watchlist/NIC industry classification and (if you want to replicate
the paper's Section 3 hedging-portfolio tests) stock return data of your
own choosing — see "What's out of scope" below.

---

## Methodology

### 1. Sentence selection (biodiversity dictionary)
Every sentence is checked against the paper's dictionary (Appendix A.4.2):
`biodiversity, ecosystem(s), ecology/ecological, habitat(s), species,
(rain)forest(s), deforestation, fauna, flora, marine, tropical, freshwater,
wetland, wildlife, coral, aquatic, desertification, carbon sink(s),
ecosphere, biosphere`.

Ambiguous unigrams are handled exactly as the paper describes, to avoid
false positives like *"software ecosystem"* or *"marine cargo insurance"*:
- **ecosystem(s) / species** only count when paired with a companion term
  anywhere in the same sentence (climate, coast, forest, nature,
  sustainability, water, ... for ecosystem; aquatic, endangered,
  habitat, protected, threatened, ... for species).
- **marine / tropical** only count through a short list of safe bigrams
  ("marine ecosystem", "tropical forest", etc.), never as bare words.

An optional **India-specific vocabulary** (toggle in the sidebar) adds
unambiguous local terms: mangroves, Western Ghats/Eastern Ghats, Sundarbans,
biodiversity hotspots, sacred groves, Ramsar wetlands, tiger reserves,
elephant corridors, wildlife sanctuaries, afforestation, agroforestry, etc.

A report **"covers biodiversity"** (Count Score = 1) if it has at least
*N* matching sentences (default N = 2, as in the paper).

### 2. Sentiment (BERT)
Each biodiversity-related sentence is classified positive / neutral /
negative by a BERT model, mirroring the paper's Section 1.2 (which uses
BERT, Devlin et al. 2018, for the same three-way classification of NYT and
10-K sentences). Two model options are offered:
- **FinBERT** (`ProsusAI/finbert`, default) — BERT-base fine-tuned on
  financial text; a natural fit for the disclosure-style language of
  annual/BRSR reports.
- **Multilingual BERT** (`nlptown/bert-base-multilingual-uncased-sentiment`)
  — a 5-star sentiment model bucketed into positive/neutral/negative,
  useful if a report mixes English with Hindi/regional-language sections.

`Negative Score = (# negative sentences) − (# positive sentences)`, and it
is 0 for reports with no biodiversity mentions or only neutral ones — same
convention as the paper's `10K-Biodiversity-Negative Score`.

### 3. Regulation score
`Regulation Score = 1` if `Count Score = 1` **and** at least one
biodiversity sentence also contains a regulation term. The dictionary
combines the paper's base terms (law, regulation, Act, discharge,
restriction) with Indian statutes/bodies: the **Wildlife Protection Act,
1972**, the **Forest (Conservation) Act, 1980**, the **Environment
(Protection) Act, 1986**, the **Biological Diversity Act, 2002**, the
**National Green Tribunal (NGT)**, **MoEFCC**, **Coastal Regulation Zone
(CRZ)** notifications, **EIA**, **CITES**, State/National Biodiversity
Boards and Authorities, and Pollution Control Boards.

### 4. Biodiversity Intensity
`= (# biodiversity sentences / # total sentences) × 100`. Not in the
original paper (which works with a large 10-K panel and mostly reports the
binary Count Score); for a single-report tool this continuous measure is
what drives the bar/line charts.

## Why annual/BRSR reports instead of 10-Ks?
India has no direct 10-K equivalent. The closest analogues are:
- **Annual reports** (mandatory for all listed companies, include a
  Management Discussion & Analysis and risk-factors-style section), and
- **BRSR (Business Responsibility & Sustainability Report)** — SEBI made
  this voluntary for the top 1,000 listed companies (by market
  capitalisation) from FY2021-22 and **mandatory from FY2022-23**, and has
  since introduced a lighter-weight **"BRSR Core"** with assurance
  requirements being phased in for the largest companies. BRSR responses
  are largely narrative/quantitative ESG disclosures, similar in spirit to
  the biodiversity-related risk-factor language the paper extracts from
  10-Ks.

Older/smaller-company reports are more likely to be scanned PDFs; see the
OCR note below.

## What's out of scope (left for you to extend)
This tool scores **one report at a time** (or a folder, via
`batch_score.py`). It does **not** replicate the paper's:
- **Aggregate news-based index** (Section 1.2: NYT/Google Trends
  biodiversity news) — you could build an Indian analogue from Indian
  newspaper archives / Google Trends India using the same
  dictionary+BERT approach in `nlp_utils.py`.
- **Industry-level aggregation and asset-pricing tests** (Sections 2.3–3):
  value-weighting firm scores up to NIC/industry level, forming long-short
  hedge portfolios, and correlating returns with news-index innovations.
  Once you have a `batch_score.py` panel across many firm-years, this is a
  standard portfolio-sorts exercise in your statistical tool of choice
  (Python/`pandas`+`statsmodels`, R, or Stata), for which you'll need
  stock-return and industry-classification data (e.g. NSE/BSE, CMIE
  Prowess, or Refinitiv) that this tool intentionally doesn't bundle.

## OCR for scanned reports (optional)
If a PDF is a scan (no selectable text), `extract_text_from_pdf` will try
an OCR fallback automatically if these are installed:
```bash
pip install pytesseract pdf2image
# plus the system binaries:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils
#   macOS:         brew install tesseract poppler
#   Windows:       install Tesseract-OCR and add it (and poppler) to PATH
```
Without these, scanned pages simply extract as empty text and won't
contribute biodiversity sentences — the app will still run, just with
lower recall on that report.

## Extending the dictionary
- Quick, one-off additions: use the **"Custom biodiversity terms"** /
  **"Custom regulation terms"** boxes in the sidebar (comma-separated).
- Permanent additions: edit `INDIA_SAFE_TERM_PATTERNS` /
  `REGULATION_TERM_PATTERNS` (or the disambiguation companion lists) in
  `biodiversity_lexicon.py`, then re-run `pytest -q` to make sure nothing
  broke.

## Citation
Giglio, S., Kuchler, T., Stroebel, J. & Zeng, X. (2023). *Biodiversity
Risk*. NBER Working Paper 31137. https://www.nber.org/papers/w31137
Public data release: https://www.biodiversityrisk.org
