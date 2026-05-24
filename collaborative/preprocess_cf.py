import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import json
import time


def process_stream_a(input_file="final_hybrid_recommender_dataset.jsonl"):
    print("🚀 Initializing Stream A (Collaborative Filtering) Preprocessing...")
    start_time = time.time()

    print(f"   -> Loading data from {input_file}...")
    df = pd.read_json(input_file, orient="records", lines=True)
    initial_len = len(df)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "parent_asin"], keep="last"
    )
    print(
        f"   -> Deduplication removed {initial_len - len(df)} duplicate interactions."
    )

    print("   -> Encoding user and item IDs...")
    user_cat = df["user_id"].astype("category")
    item_cat = df["parent_asin"].astype("category")

    df["user_index"] = user_cat.cat.codes
    df["item_index"] = item_cat.cat.codes

    num_users = len(user_cat.cat.categories)
    num_items = len(item_cat.cat.categories)
    print(
        f"   -> Matrix dimensions established: {num_users} Users x {num_items} Items."
    )

    print("   -> Constructing sparse coordinate matrix...")

    interaction_matrix = coo_matrix(
        (df["rating"], (df["user_index"], df["item_index"])),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    print("   -> Saving artifacts to disk...")

    save_npz("cf_interaction_matrix.npz", interaction_matrix)

    user_mapping = {
        int(idx): str(uid) for idx, uid in enumerate(user_cat.cat.categories)
    }
    item_mapping = {
        int(idx): str(iid) for idx, iid in enumerate(item_cat.cat.categories)
    }

    with open("cf_user_mapping.json", "w") as f:
        json.dump(user_mapping, f)

    with open("cf_item_mapping.json", "w") as f:
        json.dump(item_mapping, f)

    elapsed = round((time.time() - start_time), 2)
    print(f"🎯 Stream A complete in {elapsed} seconds!")
    print(
        "   Output files: cf_interaction_matrix.npz, cf_user_mapping.json, cf_item_mapping.json"
    )


if __name__ == "__main__":
    process_stream_a()
