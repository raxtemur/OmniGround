import json

import pandas as pd
import torch


@torch.no_grad()
def encode_image(image, clip, preprocess_function, cfg):
    image_tensor = preprocess_function(image).to(cfg.device2)
    pooling, hidden_states = clip.visual(image_tensor[None])
    return hidden_states.to(cfg.device1) # 1, 256, 1664


def calculate_recall_at_K(df, K=1, answer_column="answer", pred_column="ppls"):
    # Sort the predictions by score
    df["sorted_predictions"] = df[pred_column].apply(lambda x: sorted(x.items(), key=lambda item: item[1], reverse=False))

    # Get the top K predicted IDs
    df[f"top_{K}_predictions"] = df["sorted_predictions"].apply(lambda x: [item[0] for item in x[:K]])

    # Check if the true answer ID is in the top K predicted IDs
    df[f"is_true_in_top_{K}"] = df.apply(lambda row: row[answer_column] in row[f"top_{K}_predictions"], axis=1)

    # Calculate recall@K
    recall_at_K = df[f"is_true_in_top_{K}"].sum() / len(df)

    return df, recall_at_K


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
