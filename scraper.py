"""
scraper.py
==========
Scrapes Groww Play Store reviews naturally — no score filtering.
Fetches max volume (5000+ reviews) and lets the real distribution speak.

Output: raw_reviews.csv → columns: [score, review, date, thumbs_up]
"""

import time
import pandas as pd
from google_play_scraper import reviews, Sort

APP_ID     = "com.nextbillion.groww"
OUTPUT_CSV = "raw_reviews.csv"
LANG       = "en"
COUNTRY    = "in"
TARGET     = 6000    # overshoot so after dedup we land at 5000+
BATCH_SIZE = 200


def main():
    print("=" * 60)
    print("  Google Play Scraper — Groww (Natural Distribution)")
    print(f"  Target: {TARGET} reviews, no score filtering")
    print("=" * 60)

    all_reviews        = []
    seen_texts         = set()
    continuation_token = None
    batch_num          = 0

    # Fetch NEWEST first — most relevant for a live app sentiment story
    for sort_method, sort_name in [
        (Sort.NEWEST,       "NEWEST"),
        (Sort.MOST_RELEVANT,"MOST_RELEVANT"),
    ]:
        if len(all_reviews) >= TARGET:
            break

        print(f"\n  Fetching with sort={sort_name}...")
        continuation_token = None

        while len(all_reviews) < TARGET:
            batch_num += 1
            try:
                result, continuation_token = reviews(
                    APP_ID,
                    lang=LANG,
                    country=COUNTRY,
                    sort=sort_method,
                    count=BATCH_SIZE,
                    continuation_token=continuation_token,
                )
            except Exception as e:
                print(f"    [!] API error: {e}")
                break

            if not result:
                print(f"    [!] No more results from {sort_name}.")
                break

            new = 0
            for r in result:
                text = r.get("content", "").strip()
                if text and text not in seen_texts and len(text) > 5:
                    seen_texts.add(text)
                    all_reviews.append({
                        "score":     r.get("score"),
                        "review":    text,
                        "date":      r.get("at"),
                        "thumbs_up": r.get("thumbsUpCount", 0),
                    })
                    new += 1

            print(f"    [batch {batch_num}] +{new} new  |  total: {len(all_reviews)}")

            if continuation_token is None:
                print(f"    [!] No more pages.")
                break

            time.sleep(0.3)

    if not all_reviews:
        print("\n  No reviews scraped.")
        return

    df = pd.DataFrame(all_reviews).drop_duplicates(subset=["review"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Done! {len(df)} unique reviews saved to {OUTPUT_CSV}")
    print(f"{'=' * 60}")

    print("\n  Score distribution (REAL — no filtering):")
    score_counts = df["score"].value_counts().sort_index()
    total = len(df)
    for score, count in score_counts.items():
        bar = "█" * int(count / total * 40)
        print(f"    {score}★  {bar}  {count} ({count/total*100:.1f}%)")

    print(f"\n  Date range:")
    print(f"    Oldest: {df['date'].min()}")
    print(f"    Newest: {df['date'].max()}")


if __name__ == "__main__":
    main()