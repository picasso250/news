import csv
import os
from polymarket_utils import PolymarketAPI

def format_volume(volume):
    if volume is None:
        return "N/A"
    try:
        volume = float(volume)
        if volume >= 1_000_000:
            return f"${volume / 1_000_000:.1f}M"
        if volume >= 1_000:
            return f"${volume / 1_000:.1f}K"
        return f"${volume:.0f}"
    except (ValueError, TypeError):
        return str(volume)

def update_tsv(tsv_path):
    api = PolymarketAPI()
    rows = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='	')
        for row in reader:
            slug = row['slug']
            print(f"Updating market: {slug}...")
            market = api.get_market_by_slug(slug)
            if market:
                price = api.extract_yes_price(market)
                if price is not None:
                    row['prob'] = f"{price * 100:.1f}%"
                
                volume = market.get('volumeNum')
                if volume is not None:
                    row['cap'] = format_volume(volume)
            else:
                print(f"  Warning: Could not fetch data for {slug}")
            
            rows.append(row)
    
    # Write back to TSV
    fieldnames = ['title', 'slug', 'prob', 'cap']
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    tsv_file = 'iran-news-markets.tsv'
    update_tsv(tsv_file)
    print("Done updating markets.")
