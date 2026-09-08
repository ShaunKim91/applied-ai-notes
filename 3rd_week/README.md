# Week 3 — Multimodal Document & Image AI

Using pretrained multimodal models (instead of training your own vision
models) to turn photos, PDFs, and HTML pages into structured, trustworthy
data.

| Day | Topic | Notes |
|---|---|---|
| Day 1 | [Seeing without training: pretrained multimodal models](daily/Day1_multimodal-vision-intro.md) | CNNs conceptually, why an API call beats training a CNN, mock-first wrapper pattern |
| Day 2 | [From photo to structured JSON](daily/Day2_image-to-structured-json.md) | Prompted JSON extraction, `safe_json()` parsing, object detection contrast, PII handling |
| Day 3 | [Reading PDFs and summarizing without making things up](daily/Day3_pdf-parsing-and-summarization.md) | `pypdf` text extraction, scanned-PDF fallback, chunked map-reduce summarization, number-grounding check |
| Day 4 | [Turning a web page into a CSV report](daily/Day4_html-table-parsing-and-reporting.md) | BeautifulSoup parsing, dataframe conversion, scraping etiquette, batch error handling |

See [`concepts/3rd_week_Concepts.ipynb`](concepts/3rd_week_Concepts.ipynb) for a
single runnable notebook covering all four days with short, self-contained
examples.
