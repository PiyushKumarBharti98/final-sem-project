import pandas as pd
import numpy as np
import json
import time
from datetime import datetime


def process_stream_c(
    input_file="final_hybrid_recommender_dataset.jsonl",
    mapping_file="cf_item_mapping.json",
):
    print(" Initializing Stream C (Context & RIS) Preprocessing...")
    start_time = time.time()

    print(f"   -> Loading dataset from {input_file}...")
    df = pd.read_json(input_file, orient="records", lines=True)

    with open(mapping_file, "r") as f:
        item_mapping_rev = {v: int(k) for k, v in json.load(f).items()}

    df["categories"] = df.get("categories", "").fillna("")
    df["primary_category"] = df["categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )

    print("   -> Normalizing temporal dimensions...")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(by=["user_id", "datetime"]).reset_index(drop=True)

    print("   -> Engineering browsing sessions (48-hour threshold)...")
    df["time_diff"] = df.groupby("user_id")["datetime"].diff()

    threshold = pd.Timedelta(hours=48)
    df["is_new_session"] = (df["time_diff"] > threshold) | (df["time_diff"].isna())

    df["session_id"] = df.groupby("user_id")["is_new_session"].cumsum()

    print("   -> Calculating synthetic RIS (Recommendation Influence Score)...")

    df["prev_category"] = df.groupby("user_id")["primary_category"].shift(1)

    df["historical_categories"] = df.groupby("user_id")["primary_category"].transform(
        lambda x: [set(x.iloc[:i]) for i in range(len(x))]
    )

    def calculate_ris_heuristic(row):
        if pd.isna(row["prev_category"]):
            return 0.5

        current_cat = row["primary_category"]

        if current_cat == row["prev_category"]:
            return 0.1

        if current_cat in row["historical_categories"]:
            return 0.4

        base_ris = 0.85
        if row.get("rating", 0) >= 4.5:
            base_ris += 0.1

        return min(base_ris, 1.0)  # Cap at 1.0

    df["synthetic_RIS"] = df.apply(calculate_ris_heuristic, axis=1)

    print("   -> Compiling item sequences per user...")
    df["item_idx"] = df["parent_asin"].map(item_mapping_rev)

    session_sequences = (
        df.groupby(["user_id", "session_id"])["item_idx"].apply(list).reset_index()
    )
    session_sequences.rename(columns={"item_idx": "item_sequence"}, inplace=True)

    print("   -> Saving contextual dataset...")
    final_context_df = df[
        [
            "user_id",
            "parent_asin",
            "item_idx",
            "session_id",
            "datetime",
            "synthetic_RIS",
            "rating",
        ]
    ]

    output_file = "ctx_behavioral_data.jsonl"
    final_context_df.to_json(output_file, orient="records", lines=True)

    seq_output_file = "ctx_session_sequences.jsonl"
    session_sequences.to_json(seq_output_file, orient="records", lines=True)

    elapsed = round((time.time() - start_time), 2)
    print(f"Stream C complete in {elapsed} seconds!")
    print(f"   Output files: {output_file}, {seq_output_file}")


if __name__ == "__main__":
    process_stream_c()
