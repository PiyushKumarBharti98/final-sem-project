import pandas as pd
import numpy as np
import json
import torch
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def process_stream_b(
    input_file="final_hybrid_recommender_dataset.jsonl",
    mapping_file="cf_item_mapping.json",
):
    print("🚀 Initializing Stream B (Content-Based) Preprocessing...")
    start_time = time.time()

    print(f"   -> Loading item index mapping from {mapping_file}...")
    with open(mapping_file, "r") as f:
        item_mapping = {int(k): str(v) for k, v in json.load(f).items()}

    num_items = len(item_mapping)
    print(f"   -> Found {num_items} unique items to embed.")

    print(f"   -> Loading metadata from {input_file}...")
    columns_to_load = ["parent_asin", "title_meta", "categories", "description"]

    df = pd.read_json(input_file, orient="records", lines=True)

    df_unique_items = df.drop_duplicates(subset=["parent_asin"]).copy()

    print("   -> Synthesizing text documents for each product...")

    df_unique_items["title_meta"] = df_unique_items.get("title_meta", "").fillna("")

    def safe_join(val):
        if isinstance(val, list):
            return " ".join([str(x) for x in val])
        return str(val) if pd.notnull(val) else ""

    df_unique_items["categories_clean"] = df_unique_items.get("categories", "").apply(
        safe_join
    )
    df_unique_items["desc_clean"] = df_unique_items.get("description", "").apply(
        safe_join
    )

    # Combine everything into one rich contextual string
    df_unique_items["full_text"] = (
        df_unique_items["title_meta"]
        + " . "
        + df_unique_items["categories_clean"]
        + " . "
        + df_unique_items["desc_clean"]
    )

    text_lookup = dict(
        zip(df_unique_items["parent_asin"], df_unique_items["full_text"])
    )

    print("   -> Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> Compute device selected: {device.upper()}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("   -> Generating 384-dimensional semantic embeddings...")
    embedding_dim = 384
    item_embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)

    ordered_texts = []
    for idx in range(num_items):
        asin = item_mapping[idx]
        doc = text_lookup.get(asin, "Unknown Product")
        ordered_texts.append(doc)

    embeddings = model.encode(ordered_texts, batch_size=128, show_progress_bar=True)

    item_embeddings = np.array(embeddings, dtype=np.float32)

    output_file = "cbf_item_embeddings.npy"
    print(f"   -> Saving aligned embedding matrix to {output_file}...")
    np.save(output_file, item_embeddings)

    elapsed = round((time.time() - start_time) / 60, 2)
    print(f"🎯 Stream B complete in {elapsed} minutes!")
    print(
        f"   Output shape: {item_embeddings.shape} (Ready for PyTorch nn.Embedding.from_pretrained)"
    )


if __name__ == "__main__":
    process_stream_b()
