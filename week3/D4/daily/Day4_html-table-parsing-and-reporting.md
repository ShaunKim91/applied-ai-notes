# Day 4 — Turning a Supplier Web Page Into a CSV Report

Suppliers publish price lists as plain HTML tables. Today's task: parse the
page, pull out the table, and produce a clean CSV report — without training or
calling any AI model at all. Not every problem needs an LLM.

## Parsing HTML with BeautifulSoup

```python
from bs4 import BeautifulSoup
import requests

response = requests.get("https://example-supplier.test/catalog/fasteners")
soup = BeautifulSoup(response.text, "html.parser")

# find() returns the first match, find_all() returns every match
price_table = soup.find("table", class_="price-list")
rows = price_table.find_all("tr")

# CSS selectors are often shorter for nested structure
product_names = soup.select("table.price-list td.product-name")
```

`find`/`find_all` search by tag name and attributes; `select` uses familiar
CSS selector syntax. Both are useful — reach for whichever reads more clearly
for the structure you're navigating.

## Table rows → dataframe

Once you have the `<tr>` elements, walk each row's `<td>` cells into a list of
dicts, then load that into a dataframe:

```python
import pandas as pd

records = []
for row in rows[1:]:  # skip the header row
    cells = [td.get_text(strip=True) for td in row.find_all("td")]
    records.append({"item": cells[0], "unit_price": float(cells[1]), "stock": int(cells[2])})

df = pd.DataFrame(records)
```

From there, normal dataframe operations apply: select the columns you care
about, sort by price, and write a report.

```python
report = df[["item", "unit_price", "stock"]].sort_values("unit_price")
report.to_csv("supplier_price_report.csv", index=False)
```

## Scraping etiquette

Before pointing a scraper at any site:

- **Check `robots.txt`** (e.g. `https://example.com/robots.txt`) for paths the
  site asks crawlers not to access, and respect them.
- **Pace your requests** — add a short delay between page fetches instead of
  hammering the server in a tight loop; a `for` loop with `time.sleep(1)`
  between requests is a reasonable default for a small job.
- **Identify yourself** — set a descriptive `User-Agent` header rather than
  spoofing a browser.

## Batch error handling

When scraping many supplier pages in one run, one bad page (timeout, changed
layout, missing table) shouldn't kill the whole batch:

```python
results, failures = [], []
for url in supplier_urls:
    try:
        results.append(scrape_price_table(url))
    except Exception as exc:
        failures.append({"url": url, "error": str(exc)})

print(f"{len(results)} succeeded, {len(failures)} failed")
```

Collecting failures instead of raising immediately lets you finish the batch
and go fix the handful of problem pages afterward.

## Takeaway

HTML table scraping is a solved, non-AI problem — BeautifulSoup plus pandas
gets you from a web page to a CSV report, as long as you scrape politely and
handle per-page failures without aborting the whole run.
