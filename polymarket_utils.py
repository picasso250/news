import requests
import json
import argparse
import sys

class PolymarketAPI:
    """
    A simple wrapper for Polymarket Gamma API.
    """
    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()

    def get_market_by_slug(self, slug):
        """
        Fetch market details by slug.
        """
        url = f"{self.BASE_URL}/markets"
        params = {"slug": slug}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            markets = response.json()
            if markets and len(markets) > 0:
                return markets[0]
            return None
        except Exception as e:
            print(f"Error fetching market by slug: {e}")
            return None

    def get_active_markets(self, limit=20, offset=0):
        """
        Fetch all active markets.
        """
        url = f"{self.BASE_URL}/markets"
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset
        }
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching active markets: {e}")
            return []

    def get_markets_by_tag(self, tag_id, limit=20, offset=0):
        """
        Fetch markets by a specific tag ID.
        """
        url = f"{self.BASE_URL}/markets"
        params = {
            "tag_id": tag_id,
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset
        }
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching markets by tag {tag_id}: {e}")
            return []

    def extract_yes_price(self, market_data):
        """
        Extract the 'Yes' price (probability) from market data dictionary.
        """
        if not market_data:
            return None

        # outcomePrices and outcomes are usually stringified JSON arrays like '["0.977", "0.023"]'
        outcome_prices_raw = market_data.get("outcomePrices")
        outcomes_raw = market_data.get("outcomes")

        if outcome_prices_raw and outcomes_raw:
            try:
                outcome_prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw

                # Find the index of 'Yes'
                try:
                    yes_index = outcomes.index("Yes")
                    return float(outcome_prices[yes_index])
                except (ValueError, IndexError):
                    # Fallback to the first price
                    return float(outcome_prices[0])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback to lastTradePrice if available
        return market_data.get("lastTradePrice")

def main():
    parser = argparse.ArgumentParser(description="Polymarket API Utility CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Slug command
    slug_parser = subparsers.add_parser("slug", help="Fetch market by slug")
    slug_parser.add_argument("slug", help="The market slug")

    # Tag command
    tag_parser = subparsers.add_parser("tag", help="Fetch markets by tag ID")
    tag_parser.add_argument("tag_id", type=int, help="The tag ID")
    tag_parser.add_argument("--limit", type=int, default=10, help="Limit results (default: 10)")
    tag_parser.add_argument("--offset", type=int, default=0, help="Offset results (default: 0)")

    # Active command
    active_parser = subparsers.add_parser("active", help="Fetch active markets")
    active_parser.add_argument("--limit", type=int, default=10, help="Limit results (default: 10)")
    active_parser.add_argument("--offset", type=int, default=0, help="Offset results (default: 0)")

    args = parser.parse_args()
    api = PolymarketAPI()

    if args.command == "slug":
        market = api.get_market_by_slug(args.slug)
        if market:
            price = api.extract_yes_price(market)
            print(f"Question: {market.get('question')}")
            print(f"ID: {market.get('id')}")
            print(f"Probability (Yes): {price * 100 if price is not None else 'N/A'}%")
            print(f"Volume: {market.get('volumeNum')}")
        else:
            print(f"Market with slug '{args.slug}' not found.")

    elif args.command == "tag":
        markets = api.get_markets_by_tag(args.tag_id, limit=args.limit, offset=args.offset)
        print(f"Found {len(markets)} markets for tag ID {args.tag_id}:")
        for m in markets:
            price = api.extract_yes_price(m)
            print(f"- [{m.get('id')}] {m.get('question')} (Prob: {price * 100 if price is not None else 'N/A'}%)")

    elif args.command == "active":
        markets = api.get_active_markets(limit=args.limit, offset=args.offset)
        print(f"Found {len(markets)} active markets:")
        for m in markets:
            price = api.extract_yes_price(m)
            print(f"- [{m.get('id')}] {m.get('question')} (Prob: {price * 100 if price is not None else 'N/A'}%)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
