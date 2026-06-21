"""
Preprocessing utilities for SMS Spam Detection (Phase 01).
Used for text normalization and loading train/validation/test splits.
"""
import re
import html
import os
import pandas as pd


def normalize_text(text: str) -> str:
    """
    Normalize raw SMS text for feature extraction and modeling.
    Steps: HTML decode, lowercase, remove numbers, remove punctuation, normalize whitespace.
    """
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_splits(data_dir: str = "data/processed"):
    """
    Load train, validation, and test CSV splits from data_dir.
    Tries 'data/processed' then '../data/processed' for notebook vs project-root runs.
    Returns (df_train, df_val, df_test).
    """
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        data_dir = os.path.join("..", "data", "processed")
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return df_train, df_val, df_test


def get_X_y(df: pd.DataFrame, text_col: str = "message_normalized", label_col: str = "Class"):
    """
    Extract feature text (X) and numeric labels (y) from a split DataFrame.
    Uses label_numeric if present, else maps Class ham=0, spam=1.
    """
    X = df[text_col].fillna("").astype(str)
    if "label_numeric" in df.columns:
        y = df["label_numeric"].values
    else:
        y = df[label_col].map({"ham": 0, "spam": 1}).values
    return X, y


def load_splits_xy(data_dir: str = "data/processed", text_col: str = "message_normalized", label_col: str = "Class"):
    """
    Convenience loader used in Phase 03 notebooks.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df_train, df_val, df_test = load_splits(data_dir=data_dir)
    X_train, y_train = get_X_y(df_train, text_col=text_col, label_col=label_col)
    X_val, y_val = get_X_y(df_val, text_col=text_col, label_col=label_col)
    X_test, y_test = get_X_y(df_test, text_col=text_col, label_col=label_col)
    return X_train, X_val, X_test, y_train, y_val, y_test
