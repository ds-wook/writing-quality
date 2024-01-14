import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import q1, q3

AGGREGATIONS = [
    "count",
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "sem",
    q1,
    "median",
    q3,
    "skew",
    pd.DataFrame.kurt,
    "sum",
]


class Preprocessor:
    def __init__(self, seed):
        self.seed = seed

        self.activities = ["Input", "Remove/Cut", "Nonproduction", "Replace", "Paste"]
        self.events = [
            "q",
            "Space",
            "Backspace",
            "Shift",
            "ArrowRight",
            "Leftclick",
            "ArrowLeft",
            ".",
            ",",
            "ArrowDown",
            "ArrowUp",
            "Enter",
            "CapsLock",
            "'",
            "Delete",
            "Unidentified",
        ]
        self.text_changes = ["q", " ", "NoChange", ".", ",", "\n", "'", '"', "-", "?", ";", "=", "/", "\\", ":"]
        self.punctuations = [
            '"',
            ".",
            ",",
            "'",
            "-",
            ";",
            ":",
            "?",
            "!",
            "<",
            ">",
            "/",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "(",
            ")",
            "_",
            "+",
        ]
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]

        self.idf = defaultdict(float)

    def activity_counts(self, df):
        tmp_df = df.groupby("id").agg({"activity": list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df["activity"].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f"activity_{i}_count" for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby("id").agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f"{colname}_{i}_count" for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby("id").agg({"text_change": list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df["text_change"].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f"text_change_{i}_count" for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby("id").agg({"down_event": list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df["down_event"].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({"punct_cnt": ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df["text_change"].str.contains("=>")) & (df["text_change"] != "NoChange")].reset_index(drop=True)
        tmp_df = tmp_df.groupby("id").agg({"text_change": list}).reset_index()
        tmp_df["text_change"] = tmp_df["text_change"].apply(lambda x: "".join(x))
        tmp_df["text_change"] = tmp_df["text_change"].apply(lambda x: re.findall(r"q+", x))
        tmp_df["input_word_count"] = tmp_df["text_change"].apply(len)
        tmp_df["input_word_length_mean"] = tmp_df["text_change"].apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)
        )
        tmp_df["input_word_length_max"] = tmp_df["text_change"].apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)
        )
        tmp_df["input_word_length_std"] = tmp_df["text_change"].apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)
        )
        tmp_df.drop(["text_change"], axis=1, inplace=True)
        return tmp_df

    def make_feats(self, df):
        feats = pd.DataFrame({"id": df["id"].unique().tolist()})

        print("Engineering time data")
        for gap in self.gaps:
            df[f"up_time_shift{gap}"] = df.groupby("id")["up_time"].shift(gap)
            df[f"action_time_gap{gap}"] = df["down_time"] - df[f"up_time_shift{gap}"]
        df.drop(columns=[f"up_time_shift{gap}" for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f"cursor_position_shift{gap}"] = df.groupby("id")["cursor_position"].shift(gap)
            df[f"cursor_position_change{gap}"] = df["cursor_position"] - df[f"cursor_position_shift{gap}"]
            df[f"cursor_position_abs_change{gap}"] = np.abs(df[f"cursor_position_change{gap}"])
        df.drop(columns=[f"cursor_position_shift{gap}" for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f"word_count_shift{gap}"] = df.groupby("id")["word_count"].shift(gap)
            df[f"word_count_change{gap}"] = df["word_count"] - df[f"word_count_shift{gap}"]
            df[f"word_count_abs_change{gap}"] = np.abs(df[f"word_count_change{gap}"])
        df.drop(columns=[f"word_count_shift{gap}" for gap in self.gaps], inplace=True)

        print("Engineering statistical summaries for features")
        feats_stat = [
            ("event_id", ["max"]),
            ("up_time", ["max"]),
            ("action_time", ["max", "min", "mean", "std", "quantile", "sem", "sum", "skew", pd.DataFrame.kurt]),
            ("activity", ["nunique"]),
            ("down_event", ["nunique"]),
            ("up_event", ["nunique"]),
            ("text_change", ["nunique"]),
            ("cursor_position", ["nunique", "max", "quantile", "sem", "mean"]),
            ("word_count", ["nunique", "max", "quantile", "sem", "mean"]),
        ]
        for gap in self.gaps:
            feats_stat.extend(
                [
                    (
                        f"action_time_gap{gap}",
                        ["max", "min", "mean", "std", "quantile", "sem", "sum", "skew", pd.DataFrame.kurt],
                    ),
                    (
                        f"cursor_position_change{gap}",
                        ["max", "mean", "std", "quantile", "sem", "sum", "skew", pd.DataFrame.kurt],
                    ),
                    (
                        f"word_count_change{gap}",
                        ["max", "mean", "std", "quantile", "sem", "sum", "skew", pd.DataFrame.kurt],
                    ),
                ]
            )

        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = (
                    df.groupby(["id"])
                    .agg({colname: method})
                    .reset_index()
                    .rename(columns={colname: f"{colname}_{method_name}"})
                )
                feats = feats.merge(tmp_df, on="id", how="left")

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering event counts data")
        tmp_df = self.event_counts(df, "down_event")
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, "up_event")
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on="id", how="left")

        print("Engineering ratios data")
        feats["word_time_ratio"] = feats["word_count_max"] / feats["up_time_max"]
        feats["word_event_ratio"] = feats["word_count_max"] / feats["event_id_max"]
        feats["event_time_ratio"] = feats["event_id_max"] / feats["up_time_max"]
        feats["idle_time_ratio"] = feats["action_time_gap1_sum"] / feats["up_time_max"]

        return feats


def getEssays(df: pd.DataFrame) -> pd.DataFrame:
    textInputDf = df[["id", "activity", "cursor_position", "text_change"]]
    textInputDf = textInputDf[textInputDf.activity != "Nonproduction"]
    valCountsArr = textInputDf["id"].value_counts(sort=False).values
    lastIndex = 0
    essaySeries = pd.Series()
    for index, valCount in enumerate(valCountsArr):
        currTextInput = textInputDf[["activity", "cursor_position", "text_change"]].iloc[
            lastIndex : lastIndex + valCount
        ]
        lastIndex += valCount
        essayText = ""
        for Input in currTextInput.values:
            if Input[0] == "Replace":
                replaceTxt = Input[2].split(" => ")
                essayText = (
                    essayText[: Input[1] - len(replaceTxt[1])]
                    + replaceTxt[1]
                    + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]) :]
                )
                continue
            if Input[0] == "Paste":
                essayText = essayText[: Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]) :]
                continue
            if Input[0] == "Remove/Cut":
                essayText = essayText[: Input[1]] + essayText[Input[1] + len(Input[2]) :]
                continue
            if "M" in Input[0]:
                croppedTxt = Input[0][10:]
                splitTxt = croppedTxt.split(" To ")
                valueArr = [item.split(", ") for item in splitTxt]
                moveData = (
                    int(valueArr[0][0][1:]),
                    int(valueArr[0][1][:-1]),
                    int(valueArr[1][0][1:]),
                    int(valueArr[1][1][:-1]),
                )
                if moveData[0] != moveData[2]:
                    if moveData[0] < moveData[2]:
                        essayText = (
                            essayText[: moveData[0]]
                            + essayText[moveData[1] : moveData[3]]
                            + essayText[moveData[0] : moveData[1]]
                            + essayText[moveData[3] :]
                        )
                    else:
                        essayText = (
                            essayText[: moveData[2]]
                            + essayText[moveData[0] : moveData[1]]
                            + essayText[moveData[2] : moveData[0]]
                            + essayText[moveData[1] :]
                        )
                continue
            essayText = essayText[: Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]) :]
        essaySeries[index] = essayText
    essaySeries.index = textInputDf["id"].unique()

    return pd.DataFrame(essaySeries, columns=["essay"])


def split_essays_into_sentences(df: pd.DataFrame) -> pd.DataFrame:
    essay_df = df
    essay_df["id"] = essay_df.index
    essay_df["sent"] = essay_df["essay"].apply(lambda x: re.split("\\.|\\?|\\!", x))
    essay_df = essay_df.explode("sent")
    essay_df["sent"] = essay_df["sent"].apply(lambda x: x.replace("\n", "").strip())
    # Number of characters in sentences
    essay_df["sent_len"] = essay_df["sent"].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df["sent_word_count"] = essay_df["sent"].apply(lambda x: len(x.split(" ")))
    essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
    return essay_df


def compute_sentence_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    sent_agg_df = pd.concat(
        [
            df[["id", "sent_len"]].groupby(["id"]).agg(AGGREGATIONS),
            df[["id", "sent_word_count"]].groupby(["id"]).agg(AGGREGATIONS),
        ],
        axis=1,
    )
    sent_agg_df.columns = ["_".join(x) for x in sent_agg_df.columns]
    sent_agg_df["id"] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count": "sent_count"})
    return sent_agg_df


def split_essays_into_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    essay_df = df
    essay_df["id"] = essay_df.index
    essay_df["paragraph"] = essay_df["essay"].apply(lambda x: x.split("\n"))
    essay_df = essay_df.explode("paragraph")
    # Number of characters in paragraphs
    essay_df["paragraph_len"] = essay_df["paragraph"].apply(lambda x: len(x))
    # Number of words in paragraphs
    essay_df["paragraph_word_count"] = essay_df["paragraph"].apply(lambda x: len(x.split(" ")))
    essay_df = essay_df[essay_df.paragraph_len != 0].reset_index(drop=True)
    return essay_df


def compute_paragraph_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    paragraph_agg_df = pd.concat(
        [
            df[["id", "paragraph_len"]].groupby(["id"]).agg(AGGREGATIONS),
            df[["id", "paragraph_word_count"]].groupby(["id"]).agg(AGGREGATIONS),
        ],
        axis=1,
    )
    paragraph_agg_df.columns = ["_".join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df["id"] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count": "paragraph_count"})
    return paragraph_agg_df
