from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from featurize import (
    Preprocessor,
    compute_paragraph_aggregations,
    compute_sentence_aggregations,
    split_essays_into_paragraphs,
    split_essays_into_sentences,
)


def add_agg_features(logs: pd.DataFrame) -> pd.DataFrame:
    logs["up_time_lagged"] = logs.groupby("id")["up_time"].shift(1).fillna(logs["down_time"])
    logs["time_diff"] = abs(logs["down_time"] - logs["up_time_lagged"]) / 1000

    group = logs.groupby("id")["time_diff"]
    largest_lantency = group.max()
    smallest_lantency = group.min()
    median_lantency = group.median()
    initial_pause = logs.groupby("id")["down_time"].first() / 1000
    pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
    pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
    pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
    pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
    pauses_3_sec = group.apply(lambda x: (x > 3).sum())

    eD592674 = pd.DataFrame(
        {
            "id": logs["id"].unique(),
            "largest_lantency": largest_lantency,
            "smallest_lantency": smallest_lantency,
            "median_lantency": median_lantency,
            "initial_pause": initial_pause,
            "pauses_half_sec": pauses_half_sec,
            "pauses_1_sec": pauses_1_sec,
            "pauses_1_half_sec": pauses_1_half_sec,
            "pauses_2_sec": pauses_2_sec,
            "pauses_3_sec": pauses_3_sec,
        }
    ).reset_index(drop=True)

    return eD592674


def load_train_dataset(cfg: DictConfig) -> tuple[pd.DataFrame]:
    """Load training dataset.

    Args:
        cfg (DictConfig): Configuration.

    Returns:
        tuple[pd.DataFrame]: Training dataset.
    """

    train_logs = pd.read_csv(Path(cfg.data.path) / "train_logs.csv")
    train_scores = pd.read_csv(Path(cfg.data.path) / "train_scores.csv")
    train_essays = pd.read_csv(Path(cfg.data.path) / "train_essays_02.csv")
    train_essays.index = train_essays["Unnamed: 0"]
    train_essays.index.name = None
    train_essays.drop(columns=["Unnamed: 0"], inplace=True)

    # Sentence features for train dataset
    train_sent_df = split_essays_into_sentences(train_essays)
    train_sent_agg_df = compute_sentence_aggregations(train_sent_df)

    # Paragraph features for train dataset
    train_paragraph_df = split_essays_into_paragraphs(train_essays)
    train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)

    preprocessor = Preprocessor(seed=42)
    train_feats = preprocessor.make_feats(train_logs)

    nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
    train_feats = train_feats.drop(columns=nan_cols)

    train_agg_fe_df = train_logs.groupby("id")[
        ["down_time", "up_time", "action_time", "cursor_position", "word_count"]
    ].agg(["mean", "std", "min", "max", "last", "first", "sem", "median", "sum"])
    train_agg_fe_df.columns = ["_".join(x) for x in train_agg_fe_df.columns]
    train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
    train_agg_fe_df.reset_index(inplace=True)

    train_feats = train_feats.merge(train_agg_fe_df, on="id", how="left")
    train_eD592674 = add_agg_features(train_logs)

    train_feats = train_feats.merge(train_eD592674, on="id", how="left")
    train_feats = train_feats.merge(train_scores, on="id", how="left")
    train_feats = train_feats.merge(train_sent_agg_df, on="id", how="left")
    train_feats = train_feats.merge(train_paragraph_agg_df, on="id", how="left")
    target_col = ["score"]
    drop_cols = ["id"]
    train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]

    return train_feats[train_cols], train_feats[target_col]
