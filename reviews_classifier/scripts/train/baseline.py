from pathlib import Path

import fire
import git
import mlflow
import mlflow.sklearn
from hydra import compose, initialize_config_dir
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from reviews_classifier.data import read_jsonl

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
    if not train_path.exists():
        raise SystemExit(
            f"Processed training data not found: {str(train_path)}. Run `preprocess_data` first."
        )

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
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision_neg": precision_score(y_val, y_val_pred, pos_label=1),
            "val_recall_neg": recall_score(y_val, y_val_pred, pos_label=1),
            "val_f1_neg": f1_score(y_val, y_val_pred, pos_label=1),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipe, name="tfidf_logreg_pipeline")

        disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
        fig = disp.figure_
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)


def train_baseline(overrides: list[str] | None = None) -> None:
    cfg = get_cfg(overrides)
    _train(cfg)


def main() -> None:
    fire.Fire(train_baseline)


if __name__ == "__main__":
    main()
