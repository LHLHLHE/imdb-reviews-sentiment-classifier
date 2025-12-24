from pathlib import Path

import fire
import git
import mlflow
import mlflow.sklearn
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from reviews_classifier.data import read_jsonl
from reviews_classifier.utils import check_data_exists

ROOT_DIR = Path(__file__).resolve().parents[3]
HYDRA_CONFIG_DIR = ROOT_DIR / "configs"


def get_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def _train(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    model_cfg = cfg.model
    split_cfg = cfg.train.split
    logging_cfg = cfg.logging

    train_path = ROOT_DIR / data_cfg.preprocess.processed_train_data_path
    test_path = ROOT_DIR / data_cfg.preprocess.processed_test_data_path

    data_exists, missing_str = check_data_exists(train_path, test_path)
    if not data_exists:
        raise SystemExit(f"Processed data not found: {missing_str}. Run `preprocess_data` first.")

    text_key = data_cfg.preprocess.text_field
    label_key = data_cfg.preprocess.label_field

    mlflow.set_tracking_uri(logging_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(logging_cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="baseline"):
        commit = git.Repo(str(ROOT_DIR)).head.commit.hexsha
        mlflow.log_params(
            {
                "model_type": "baseline",
                "git_commit": commit,
                "val_size": float(split_cfg.val_size),
                "random_state": int(split_cfg.random_state),
                "C": float(model_cfg.C),
                "max_iter": int(model_cfg.max_iter),
                "class_weight": str(model_cfg.class_weight),
                "shuffle": bool(split_cfg.shuffle),
                "metric_pos_label": 1,
            }
        )
        mlflow.set_tag("git_commit", commit)

        X_text, y = read_jsonl(train_path, text_key, label_key)

        X_train, X_val, y_train, y_val = train_test_split(
            X_text,
            y,
            test_size=split_cfg.val_size,
            random_state=split_cfg.random_state,
            shuffle=split_cfg.shuffle,
            stratify=y,
        )

        tfidf = TfidfVectorizer()
        logreg = LogisticRegression(
            C=model_cfg.C, max_iter=model_cfg.max_iter, class_weight=model_cfg.class_weight
        )
        pipe = Pipeline([("tfidf", tfidf), ("logreg", logreg)])
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)
        metrics = {
            "train/accuracy": accuracy_score(y_train, y_train_pred),
            "val/accuracy": accuracy_score(y_val, y_val_pred),
            "val/precision_neg": precision_score(y_val, y_val_pred, pos_label=1),
            "val/recall_neg": recall_score(y_val, y_val_pred, pos_label=1),
            "val/f1_neg": f1_score(y_val, y_val_pred, pos_label=1),
        }
        mlflow.log_metrics(metrics)

        X_test, y_test = read_jsonl(test_path, text_key, label_key)
        y_test_pred = pipe.predict(X_test)
        test_metrics = {
            "test/accuracy": accuracy_score(y_test, y_test_pred),
            "test/precision_neg": precision_score(y_test, y_test_pred, pos_label=1),
            "test/recall_neg": recall_score(y_test, y_test_pred, pos_label=1),
            "test/f1_neg": f1_score(y_test, y_test_pred, pos_label=1),
        }
        mlflow.log_metrics(test_metrics)

        test_metrics_str = "\n\t".join(
            f"{metric}: {value}" for metric, value in test_metrics.items()
        )
        print(f"Test metrics:\n\t{test_metrics_str}")

        mlflow.sklearn.log_model(pipe, name="tfidf_logreg_pipeline")


def train_baseline(overrides: list[str] | None = None) -> None:
    cfg = get_cfg(overrides)
    _train(cfg)


def main() -> None:
    fire.Fire(train_baseline)


if __name__ == "__main__":
    main()
