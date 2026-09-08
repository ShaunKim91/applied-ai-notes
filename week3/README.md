# Week 3 — Multimodal Document & Image AI

Using pretrained multimodal models (instead of training your own vision
models) to turn photos, PDFs, and HTML pages into structured, trustworthy
data.

| Day | Topic | Notes |
|---|---|---|
| D1 | [Seeing without training: pretrained multimodal models](D1/daily/Day1_multimodal-vision-intro.md) | CNNs conceptually, why an API call beats training a CNN, mock-first wrapper pattern |
| D2 | [From photo to structured JSON](D2/daily/Day2_image-to-structured-json.md) | Prompted JSON extraction, `safe_json()` parsing, object detection contrast, PII handling |
| D3 | [Reading PDFs and summarizing without making things up](D3/daily/Day3_pdf-parsing-and-summarization.md) | `pypdf` text extraction, scanned-PDF fallback, chunked map-reduce summarization, number-grounding check |
| D4 | [Turning a web page into a CSV report](D4/daily/Day4_html-table-parsing-and-reporting.md) | BeautifulSoup parsing, dataframe conversion, scraping etiquette, batch error handling |

See [`concepts/Week3_Concepts.ipynb`](concepts/Week3_Concepts.ipynb) for a
single runnable notebook covering all four days with short, self-contained
examples.
