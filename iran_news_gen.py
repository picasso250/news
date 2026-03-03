import markdown
import csv
from bs4 import BeautifulSoup
import os
import re

def generate_timeline_html(tsv_path):
    timeline_items = []
    if not os.path.exists(tsv_path):
        return f"<!-- File not found: {tsv_path} -->"
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='	')
        for row in reader:
            item = f"""
     <div class="timeline-item">
      <div class="timestamp-group">
       <span class="timestamp">北京: {row['beijing_time']}</span>
       <span class="timestamp">当地: {row['local_time']}</span>
       <span class="timestamp">UTC: {row['utc']}</span>
      </div>
      <div class="event-content">
       {row['text']}
      </div>
     </div>"""
            timeline_items.append(item)
    
    return '<div class="timeline">' + "".join(timeline_items) + '</div>'

def generate_markets_html(tsv_path):
    if not os.path.exists(tsv_path):
        return f"<!-- File not found: {tsv_path} -->"
    
    rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='	')
        for row in reader:
            url = f"https://polymarket.com/event/{row['slug']}"
            rows.append(f"""
      <tr>
       <td><a href="{url}">{row['title']}</a></td>
       <td>{row['prob']}</td>
       <td>{row['cap']}</td>
      </tr>""")
    
    html = f"""
    <table class="timezone-table">
     <thead>
      <tr>
       <th>预测问题</th>
       <th>概率 (Yes)</th>
       <th>交易量</th>
      </tr>
     </thead>
     <tbody>
      {"".join(rows)}
     </tbody>
    </table>"""
    return html

def handle_insert(match):
    label = match.group(1)
    path = match.group(2)
    
    if "timeline" in path:
        return f"<!-- INSERT_START:{path} -->\n{generate_timeline_html(path)}\n<!-- INSERT_END -->"
    elif "markets" in path:
        return f"<!-- INSERT_START:{path} -->\n{generate_markets_html(path)}\n<!-- INSERT_END -->"
    else:
        return f"<!-- Unsupported insert: {path} -->"

def convert_md_to_html(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Use regex to find [INSERT label](path) and replace it with HTML placeholders
    # We do this before MD conversion to ensure our custom HTML is preserved or 
    # handled correctly.
    md_content = re.sub(r'\[INSERT .*?\]\((.*?)\)', lambda m: f"<!-- DIV_START:{m.group(1)} -->", md_content)
    # Also handle the old MARKETS/TIMELINE just in case
    md_content = md_content.replace('TIMELINE', '').replace('MARKETS', '')
    
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    soup = BeautifulSoup(html_body, 'html.parser')
    
    # Now find the placeholders in the soup and replace them with actual data
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and "DIV_START:" in text):
        path = comment.split("DIV_START:")[1].strip()
        if "timeline" in path:
            new_content = BeautifulSoup(generate_timeline_html(path), 'html.parser')
        elif "markets" in path:
            new_content = BeautifulSoup(generate_markets_html(path), 'html.parser')
        else:
            new_content = BeautifulSoup(f"<p>Unsupported: {path}</p>", 'html.parser')
        
        comment.replace_with(new_content)

    # Process tables
    tables = soup.find_all('table')
    for i, table in enumerate(tables):
        if i == 0:
            table['class'] = 'timezone-table'
        else:
            if not table.get('class'):
                table['class'] = 'fact-check-table'

    return str(soup)

def main():
    md_file = 'iran-news-zh.md'
    output_file = 'iran-news-zh.html'
    
    content_html = convert_md_to_html(md_file)
    
    template = f"""<!DOCTYPE html>
<html lang="zh-CN">
 <head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>伊朗最新新闻汇总 - 2026年3月1日</title>
  <style>
   :root {{
            --primary-color: #e63946;
            --secondary-color: #457b9d;
            --bg-color: #f1faee;
            --dark-bg: #1d3557;
            --text-color: #1d3557;
            --light-text: #f1faee;
            --accent-color: #a8dadc;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
            background-color: #f8f9fa;
            color: var(--text-color);
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 4px solid var(--primary-color);
            padding-bottom: 20px;
        }}
        h1 {{
            font-size: 2.5em;
            color: var(--dark-bg);
            margin-bottom: 10px;
        }}
        .update-time {{
            color: #666;
            font-size: 0.9em;
        }}
        .breaking-banner {{
            background-color: var(--primary-color);
            color: white;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        h2 {{
            color: var(--dark-bg);
            margin-top: 30px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent-color);
            display: flex;
            align-items: center;
        }}
        h2::before {{
            content: "";
            display: inline-block;
            width: 10px;
            height: 24px;
            background-color: var(--primary-color);
            margin-right: 10px;
        }}
        h3 {{
            color: var(--secondary-color);
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        /* Timezone Table */
        .timezone-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .timezone-table th, .timezone-table td {{
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: center;
        }}
        .timezone-table th {{
            background-color: var(--dark-bg);
            color: white;
        }}
        /* Timeline */
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        .timeline-item {{
            padding: 20px;
            border-left: 2px solid var(--primary-color);
            position: relative;
            margin-left: 20px;
            background: #fdfdfd;
            margin-bottom: 15px;
            border-radius: 0 4px 4px 0;
        }}
        .timeline-item::before {{
            content: "";
            position: absolute;
            left: -9px;
            top: 24px;
            width: 16px;
            height: 16px;
            background-color: white;
            border: 2px solid var(--primary-color);
            border-radius: 50%;
        }}
        .timestamp-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 10px;
            font-size: 0.85em;
        }}
        .timestamp {{
            background: var(--accent-color);
            padding: 2px 8px;
            border-radius: 4px;
            color: var(--dark-bg);
            font-weight: bold;
        }}
        .event-content {{
            font-size: 1.05em;
        }}
        /* Fact Check Table */
        .fact-check-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 20px;
        }}
        .fact-check-table th, .fact-check-table td {{
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }}
        .fact-check-table th {{
            background-color: #f8f9fa;
            width: 30%;
            color: #495057;
        }}
        .source-link {{
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: bold;
        }}
        .source-link:hover {{
            text-decoration: underline;
        }}
        footer {{
            text-align: center;
            margin-top: 50px;
            color: #6c757d;
            font-size: 0.9em;
            padding-bottom: 40px;
        }}
        @media (max-width: 600px) {{
            .timestamp-group {{
                flex-direction: column;
                gap: 5px;
            }}
        }}
        /* Markdown overrides */
        blockquote {{
            border-left: 5px solid var(--accent-color);
            padding-left: 15px;
            color: #666;
            margin: 20px 0;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #eee;
            margin: 40px 0;
        }}
  </style>
 </head>
 <body>
  <div class="container">
   <header>
    <h1>伊朗最新新闻汇总</h1>
    <p class="update-time">最后更新：2026年3月1日 11:35 (北京时间)</p>
   </header>
   <div class="breaking-banner">
    实时更新：伊朗国家电视台正式确认哈梅内伊在美以空袭中身亡，伊朗成立权力过渡委员会
   </div>
   <div class="content-body">
    {content_html}
   </div>
   <footer>
    <p>© 2026 新闻汇总 | 仅供信息参考</p>
    <p>本页所有信息均经多信源核实，由于局势瞬息万变，请关注最新官方通告。</p>
   </footer>
  </div>
 </body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    main()
