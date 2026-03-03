# Project TODOs

- [ ] Explore more idiomatic Markdown extensions or custom syntax (e.g., `[[TIMELINE]]` or specialized blocks) to replace the `TIMELINE` placeholder.
- [ ] Generalize the rendering script to support multiple "blueprints" or page types.
- [ ] Implement a unified "render blueprint" configuration to manage data-to-HTML mappings.
- [ ] Set up a Render Deploy Hook and update the Cron Job to trigger a site rebuild after data updates.
- [ ] Implement an hourly Cron Job to fetch latest Polymarket data (using `polymarket_utils.py`) to keep probabilities fresh.
