"""
Lightweight inference helper for the SMS spam project.
Provides functions to load saved artifacts from `results/model_artifacts/` and run
predictions on single texts or batches.
"""
import os
import joblib
import pandas as pd
from typing import List
from src.preprocessing import normalize_text

MODEL_DIR = os.path.join("results", "model_artifacts")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "production_model.joblib")
DEFAULT_VECT = os.path.join(MODEL_DIR, "phase4_vectorizer.joblib")




def load_artifacts(model_path: str = None, vectorizer_path: str = None):
    """Load and return (model, vectorizer). Paths default to `results/model_artifacts`."""
    model_path = model_path or DEFAULT_MODEL
    vectorizer_path = vectorizer_path or DEFAULT_VECT
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model or vectorizer not found in {MODEL_DIR}")
    model = joblib.load(model_path)
    vec = joblib.load(vectorizer_path)
    return model, vec

def predict_texts(texts: List[str], model=None, vec=None) -> pd.DataFrame:
    if model is None:
        model, _ = load_artifacts()

    texts_norm = [normalize_text(t) for t in texts]

    preds = model.predict(texts_norm)

    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(texts_norm)
        else:
            scores = model.predict_proba(texts_norm)[:, 1]
    except Exception:
        scores = [None] * len(preds)

    df = pd.DataFrame({
        "text": texts,
        "text_normalized": texts_norm,
        "pred_label": preds,
        "score": scores,
    })
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with saved model and vectorizer")
    parser.add_argument("--text", type=str, help="Single text to predict (overrides --input)")
    parser.add_argument("--input", type=str, help="Path to a newline-delimited text file with examples")
    parser.add_argument("--model", type=str, help="Path to model.joblib (optional)")
    parser.add_argument("--vectorizer", type=str, help="Path to vectorizer.joblib (optional)")
    parser.add_argument("--out", type=str, default="results/model_artifacts/predictions_from_cli.csv", help="CSV output path")
    args = parser.parse_args()

    if args.text:
        df_out = predict_texts([args.text], model=None, vec=None)
        print(df_out.to_dict(orient="records"))
    elif args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(args.input)
        with open(args.input, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        df_out = predict_texts(lines)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df_out.to_csv(args.out, index=False)
        print(f"Wrote predictions to {args.out}")
    else:
        parser.print_help()
