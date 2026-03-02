import json
import os
import subprocess
import requests
from bs4 import BeautifulSoup
import urllib.parse
from datetime import datetime
import re

CONFIG_FILE = "rss_feeds.json"
DOWNLOADED_FILE = "downloaded.json"
ARTICLES_DIR = "articles"

def load_json(filepath, default_val):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_val
    return default_val

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_filename(title):
    # Remove invalid characters
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title).strip()
    # Replace spaces with hyphens
    return re.sub(r'\s+', "-", cleaned)

def fetch_feed(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error fetching feed {url}: {e}")
        return None

def parse_feed(content):
    if not content:
        return []

    soup = BeautifulSoup(content, 'xml')
    entries = []

    items = soup.find_all('item')
    if items:
        for item in items:
            title = item.title.text if item.title else "Untitled"
            link = item.link.text if item.link else ""
            pub_date = item.pubDate.text if item.pubDate else ""
            entries.append({'title': title, 'link': link, 'date': pub_date})
        return entries

    entries_atom = soup.find_all('entry')
    if entries_atom:
        for entry in entries_atom:
            title = entry.title.text if entry.title else "Untitled"
            link = ""
            link_tag = entry.find('link', rel='alternate') or entry.find('link')
            if link_tag and link_tag.has_attr('href'):
                link = link_tag['href']
            elif entry.id:
                link = entry.id.text

            pub_date = entry.published.text if entry.published else (entry.updated.text if entry.updated else "")
            entries.append({'title': title, 'link': link, 'date': pub_date})

    return entries

def main():
    feeds = load_json(CONFIG_FILE, [])
    if not feeds:
        print(f"No feeds found in {CONFIG_FILE}.")
        return

    downloaded = load_json(DOWNLOADED_FILE, {})

    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)

    all_articles = []
    new_downloads_count = 0

    for feed in feeds:
        feed_name = feed.get('name', 'Unknown')
        feed_url = feed.get('url')
        print(f"\nProcessing feed: {feed_name}")

        feed_dir = os.path.join(ARTICLES_DIR, clean_filename(feed_name))
        if not os.path.exists(feed_dir):
            os.makedirs(feed_dir)

        content = fetch_feed(feed_url)
        entries = parse_feed(content)

        for entry in entries:
            link = entry['link']
            title = entry['title']
            date_str = entry['date']

            if link and not link.startswith('http'):
                link = urllib.parse.urljoin(feed_url, link)

            if not link:
                continue

            all_articles.append({
                'feed': feed_name,
                'title': title,
                'link': link,
                'date': date_str
            })

            if link in downloaded:
                continue

            print(f"  New article found: {title}")

            safe_title = clean_filename(title)
            date_prefix = ""
            date_sortable = ""

            # extract year-month-day
            match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', date_str)
            if match:
                date_prefix = match.group(1).replace('/', '-') + "_"
                date_sortable = match.group(1).replace('/', '-')
            else:
                try:
                    # check for common text date formats
                    parsed_date = datetime.strptime(date_str[:16], "%a, %d %b %Y")
                    date_prefix = parsed_date.strftime("%Y-%m-%d_")
                    date_sortable = parsed_date.strftime("%Y-%m-%d")
                except:
                    pass

            filename = f"{date_prefix}{safe_title}.md"
            filepath = os.path.join(feed_dir, filename)

            print(f"  Downloading to {filepath}...")

            cmd = ["python3", "url2md.py", "--url", link, "--out-file", filepath]
            try:
                # To prevent blocking, we limit processing time
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
                downloaded[link] = {
                    'title': title,
                    'date': date_str,
                    'date_sortable': date_sortable,
                    'filepath': filepath,
                    'feed': feed_name
                }
                new_downloads_count += 1
                save_json(DOWNLOADED_FILE, downloaded)
                print(f"  Success.")
            except subprocess.TimeoutExpired:
                 print(f"  Failed: Timeout running url2md.py")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to download: {e.stderr}")
            except Exception as e:
                print(f"  Error running url2md.py: {e}")

    print(f"\nDownloaded {new_downloads_count} new articles.")
    generate_index(all_articles, downloaded)

def generate_index(all_articles, downloaded):
    index_path = os.path.join(ARTICLES_DIR, "index.md")
    print("Generating index.md...")

    # Sort properly by date_sortable (ISO format string), fallback to date_str if empty
    sorted_downloads = sorted(
        downloaded.values(),
        key=lambda x: x.get('date_sortable') or x.get('date', ''),
        reverse=True
    )

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# RSS Reader Index\n\n")
        f.write("Welcome to your local RSS Reader index. Below are the downloaded articles.\n\n")

        f.write("## All Downloaded Articles\n\n")
        for item in sorted_downloads:
            feed_name = item.get('feed', 'Unknown')
            title = item.get('title', 'Untitled')
            date_str = item.get('date', '')
            filepath = item.get('filepath', '')

            rel_path = os.path.relpath(filepath, ARTICLES_DIR)

            f.write(f"- **{feed_name}**: [{title}](./{urllib.parse.quote(rel_path)}) *({date_str})*\n")

    print("Done!")

if __name__ == "__main__":
    main()
