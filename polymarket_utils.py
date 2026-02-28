import requests
import json

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

    def extract_yes_price(self, market_data):
        """
        Extract the 'Yes' price (probability) from market data dictionary.
        """
        if not market_data:
            return None

        # outcomePrices and outcomes are usually stringified JSON arrays like '["0.977", "0.023"]'
        # but sometimes they might already be parsed depending on the API client or gateway
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
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                pass

        # Fallback to lastTradePrice if available
        return market_data.get("lastTradePrice")

    def get_market_price(self, slug):
        """
        Convenience method to get price directly from slug.
        Note: If you already have market data, use extract_yes_price(market_data) to avoid an extra API call.
        """
        market = self.get_market_by_slug(slug)
        return self.extract_yes_price(market)

if __name__ == "__main__":
    api = PolymarketAPI()
    target_slug = "khamenei-out-as-supreme-leader-of-iran-by-march-31"

    print(f"Fetching details for slug: {target_slug}...")
    market_details = api.get_market_by_slug(target_slug)

    if market_details:
        print("\nMarket Details:")
        print(f"Question: {market_details.get('question')}")
        print(f"ID: {market_details.get('id')}")
        print(f"Ends: {market_details.get('endDate')}")

        # Extract price from the already fetched market_details to avoid redundant call
        price = api.extract_yes_price(market_details)
        print(f"\nLatest 'Yes' Price (Probability): {price}")

        if price is not None:
            print(f"Verification successful: Current probability is {price * 100:.2f}%")
        else:
            print("Verification failed: Could not retrieve price.")
    else:
        print("Verification failed: Could not retrieve market details.")
