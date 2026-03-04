import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


LABEL_START_INDEX = 6


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    label_columns = list(df.columns[LABEL_START_INDEX:])
    y = df[label_columns].to_numpy(dtype=np.int8)
    return df, y, label_columns


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        C=4.0,
                    ),
                    n_jobs=None,
                ),
            ),
        ]
    )


def get_label_scores(model: Pipeline, texts: pd.Series) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(texts)

    scores = model.decision_function(texts)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (scores >= threshold).astype(np.int8)
        metrics = evaluate_predictions(y_true, y_pred)
        if best_metrics is None or metrics["micro_f1"] > best_metrics["micro_f1"]:
            best_threshold = float(round(threshold, 2))
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def decode_labels(binary_matrix: np.ndarray, label_columns: list[str]) -> list[list[str]]:
    decoded: list[list[str]] = []
    for row in binary_matrix:
        decoded.append([label for label, flag in zip(label_columns, row) if flag])
    return decoded


def save_predictions(
    df: pd.DataFrame,
    scores: np.ndarray,
    y_pred: np.ndarray,
    label_columns: list[str],
    output_path: Path,
) -> None:
    out_df = df[["text"]].copy()
    out_df["predicted_labels"] = decode_labels(y_pred, label_columns)
    out_df["predicted_label_count"] = y_pred.sum(axis=1)

    top_k = np.argsort(-scores, axis=1)[:, :5]
    top_labels: list[list[str]] = []
    top_scores: list[list[float]] = []
    for row_index, indices in enumerate(top_k):
        top_labels.append([label_columns[i] for i in indices])
        top_scores.append([round(float(scores[row_index, i]), 4) for i in indices])

    out_df["top5_labels"] = top_labels
    out_df["top5_scores"] = top_scores

    if "labels" in df.columns:
        out_df["gold_labels"] = df["labels"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_pipeline(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    train_df, y_train, label_columns = load_dataset(train_path)
    valid_df, y_valid, valid_label_columns = load_dataset(valid_path)
    test_df, y_test, test_label_columns = load_dataset(test_path)

    if label_columns != valid_label_columns or label_columns != test_label_columns:
        raise ValueError("Train/valid/test label columns are not aligned.")

    model = build_model()
    model.fit(train_df["text"], y_train)

    valid_scores = get_label_scores(model, valid_df["text"])
    best_threshold, valid_metrics = tune_threshold(y_valid, valid_scores)

    test_scores = get_label_scores(model, test_df["text"])
    test_pred = (test_scores >= best_threshold).astype(np.int8)
    test_metrics = evaluate_predictions(y_test, test_pred)

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "label_columns": label_columns,
                "threshold": best_threshold,
            },
            file,
        )

    metrics = {
        "threshold": best_threshold,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "label_count": int(len(label_columns)),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    save_predictions(
        df=test_df,
        scores=test_scores,
        y_pred=test_pred,
        label_columns=label_columns,
        output_path=output_dir / "test_predictions.csv",
    )

    return metrics


def predict_text(model_path: Path, text: str, top_k: int) -> dict[str, object]:
    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    model: Pipeline = bundle["model"]
    label_columns: list[str] = bundle["label_columns"]
    threshold: float = bundle["threshold"]

    scores = get_label_scores(model, pd.Series([text]))
    score_row = scores[0]
    pred = (score_row >= threshold).astype(np.int8)

    ranked = np.argsort(-score_row)[:top_k]
    return {
        "threshold": threshold,
        "predicted_labels": [label_columns[i] for i, flag in enumerate(pred) if flag],
        "top_labels": [label_columns[i] for i in ranked],
        "top_scores": [round(float(score_row[i]), 4) for i in ranked],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a multi-label text model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train, infer on test set, and evaluate.")
    train_parser.add_argument("--train", default="data/train_multilabel.csv")
    train_parser.add_argument("--valid", default="data/valid_multilabel.csv")
    train_parser.add_argument("--test", default="data/test_multilabel.csv")
    train_parser.add_argument("--output-dir", default="artifacts/multilabel_baseline")

    predict_parser = subparsers.add_parser("predict-text", help="Run inference for a single text.")
    predict_parser.add_argument("--model-path", default="artifacts/multilabel_baseline/model.pkl")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        metrics = run_pipeline(
            train_path=Path(args.train),
            valid_path=Path(args.valid),
            test_path=Path(args.test),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "predict-text":
        result = predict_text(
            model_path=Path(args.model_path),
            text=args.text,
            top_k=args.top_k,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
