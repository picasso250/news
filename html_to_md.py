import argparse
import asyncio
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, NavigableString, Tag

async def get_html(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Set a timeout and wait for network idle to ensure dynamic content is loaded
        await page.goto(url, wait_until="networkidle", timeout=60000)
        content = await page.content()
        await browser.close()
        return content

def convert_to_md(soup):
    # Remove script and style elements as they don't contain user-visible text
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    def walk(element):
        if isinstance(element, NavigableString):
            return element.strip()

        if not isinstance(element, Tag):
            return ""

        tag_name = element.name

        # 1. Handle headers h1-h4
        if tag_name in ['h1', 'h2', 'h3', 'h4']:
            level = int(tag_name[1])
            inner = " ".join(filter(None, [walk(c) for c in element.children]))
            return f"\n{'#' * level} {inner}\n"

        # 2. Handle links (a tags)
        if tag_name == 'a':
            href = element.get('href', '')
            inner = " ".join(filter(None, [walk(c) for c in element.children]))
            if href and inner:
                return f"[{inner}]({href})"
            return inner

        # 3. Handle paragraphs (p tags)
        if tag_name == 'p':
            inner = " ".join(filter(None, [walk(c) for c in element.children]))
            return f"\n{inner}\n"

        # 4. For all other tags, extract text from children and join with space
        return " ".join(filter(None, [walk(c) for c in element.children]))

    target = soup.body if soup.body else soup
    md_content = walk(target)

    # Post-processing:
    # Remove multiple spaces within a line
    md_content = re.sub(r' +', ' ', md_content)

    # Split by newline, strip each line, and remove empty lines to handle "p标签之间使用换行"
    # and general formatting requirements.
    lines = [line.strip() for line in md_content.split('\n')]
    return "\n".join(filter(None, lines))

async def main():
    parser = argparse.ArgumentParser(description="Convert HTML from URL to Markdown")
    parser.add_argument("--url", required=True, help="URL to fetch")
    parser.add_argument("--out-file", required=True, help="Output markdown file")

    args = parser.parse_args()

    try:
        print(f"Fetching {args.url}...")
        html = await get_html(args.url)

        print("Parsing HTML...")
        # Use lxml if available, otherwise fallback to html.parser
        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')

        print("Converting to Markdown...")
        markdown = convert_to_md(soup)

        with open(args.out_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"Successfully saved to {args.out_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
