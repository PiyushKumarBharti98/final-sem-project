import os
import pandas as pd
from datasets import load_dataset
import requests
import gzip
import json
import time

categories = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]

sample_size = 3000
checkpoint_file = "checkpoint_recommender_data.jsonl"
final_file = "final_hybrid_recommender_dataset.jsonl"
all_merged_dfs = []
start_time = time.time()

if os.path.exists(checkpoint_file):
    print(" Found an old checkpoint file. Deleting it for a fresh run...")
    os.remove(checkpoint_file)

start_time = time.time()
print(f" Starting resilient extraction pipeline for {len(categories)} categories...")

print(f" Starting hybrid extraction pipeline for {len(categories)} categories...")

for idx, category in enumerate(categories, 1):
    print(f"\n[{idx}/{len(categories)}]  Processing: {category}")

    try:
        print("   -> Fetching reviews via Hugging Face...")
        reviews_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            streaming=True,
            trust_remote_code=True,
        )

        reviews_df = pd.DataFrame(list(reviews_dataset.take(sample_size)))
        product_ids = set(reviews_df["parent_asin"])

        print(
            f"   -> Found {len(product_ids)} unique products. Streaming metadata natively..."
        )

        meta_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"
        meta_records = []

        with requests.get(meta_url, stream=True) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if line:
                    item = json.loads(line)
                    if item.get("parent_asin") in product_ids:
                        meta_records.append(item)

                    if len(meta_records) == len(product_ids):
                        break

        meta_df = pd.DataFrame(meta_records)
        print(f"   -> Extracted {len(meta_df)} matching metadata records.")

        merged_df = pd.merge(
            reviews_df,
            meta_df,
            on="parent_asin",
            how="left",
            suffixes=("_review", "_meta"),
        )

        merged_df["source_category"] = category

        print(f"    Appending {len(merged_df)} rows to checkpoint file...")
        merged_df.to_json(checkpoint_file, orient="records", lines=True, mode="a")

        all_merged_dfs.append(merged_df)
        print("   Merge successful.")

    except Exception as e:
        print(f"    Error processing {category}: {e}")
        print("   Moving to the next category. Previous data is safe on disk.")

if os.path.exists(checkpoint_file):
    print("\n Pipeline complete. Reading checkpoint data for final shuffle...")

    final_df = pd.read_json(checkpoint_file, orient="records", lines=True)

    print(" Randomizing the entire dataset to prevent sequence bias...")
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f" Saving randomized final dataset to: {final_file}")
    final_df.to_json(final_file, orient="records", lines=True)

    elapsed = round((time.time() - start_time) / 60, 2)
    print(f" Complete! Processed {len(final_df)} total rows in {elapsed} minutes.")
else:
    print(" Pipeline failed entirely. No data was processed.")
