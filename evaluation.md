# Evaluation: `html_to_md.py` vs. `view_text_website`

This document compares the performance and output quality of the custom `html_to_md.py` script and the built-in `view_text_website` tool.

## Comparison Table

| Feature | `html_to_md.py` | `view_text_website` |
| :--- | :--- | :--- |
| **Technology** | Playwright (Chromium) + BeautifulSoup4 | Internal Trawler/Fetch service |
| **Output Format** | Structured Markdown (h1-h4, links) | Plain text with numbered references |
| **Dynamic Content** | **Excellent.** Handles SPA, React, and slow-loading elements via `networkidle`. | **Basic.** Often fails on JavaScript-heavy sites. |
| **Header Handling** | Preserves `h1`-`h4` as Markdown headers. | Renders headers as capitalized or bold text, no Markdown syntax. |
| **Link Handling** | Inline Markdown links: `[text](url)`. | Numbered references: `[1]text` with a footer. |
| **Whitespace** | Controlled: spaces between elements, newlines between `p` tags. | Basic whitespace preservation. |
| **Bypass Capabilities**| **Stronger.** Playwright emulates a real browser, often bypassing basic bot detection. | **Weaker.** Frequently rejected by major sites (CNN, BBC). |
| **Customizability** | Fully customizable logic for tag processing. | Fixed internal logic. |

## Key Findings

### 1. Robustness against Bot Detection
During testing on **CNN**, **BBC**, and **Reuters**, `view_text_website` failed with `REJECTED_CLIENT_CAPABILITY`. In contrast, `html_to_md.py` (using Playwright) successfully fetched and parsed all three sites.

### 2. Output Structure
*   `html_to_md.py` produces a "cleaner" Markdown that is directly usable in documentation or further processing. It strictly follows the requirement to only format headers and links, reducing "noise" while keeping text content.
*   `view_text_website` provides a more "rendered" text view, which can sometimes be harder to parse programmatically due to the reference list at the bottom.

### 3. Handling of Modern Web Apps
Many news sites use complex JavaScript. `html_to_md.py` waits for the network to be idle, ensuring that headlines and article bodies are fully rendered before conversion. `view_text_website` often misses content that isn't in the initial static HTML.

## Conclusion

The custom `html_to_md.py` script is significantly more powerful for scraping modern, JavaScript-heavy websites like major news outlets. Its ability to produce standard Markdown headers and inline links makes it superior for tasks requiring structured data extraction, whereas `view_text_website` is a quick tool for simple, static pages.
