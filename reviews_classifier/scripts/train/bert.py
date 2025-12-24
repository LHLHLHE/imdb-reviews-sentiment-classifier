from pathlib import Path

import fire
import git
import lightning as L
import mlflow
import mlflow.pytorch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from reviews_classifier.data import ImdbReviewsDataModule, SplitConfig
from reviews_classifier.module import BertSentimentModule
from reviews_classifier.utils import check_data_exists

ROOT_DIR = Path(__file__).resolve().parents[3]
HYDRA_CONFIG_DIR = ROOT_DIR / "configs"


def get_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def _train(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    split_cfg = train_cfg.split
    logging_cfg = cfg.logging

    train_path = ROOT_DIR / data_cfg.preprocess.processed_train_data_path
    test_path = ROOT_DIR / data_cfg.preprocess.processed_test_data_path

    data_exists, missing_str = check_data_exists(train_path, test_path)
    if not data_exists:
        raise SystemExit(f"Processed data not found: {missing_str}. Run `preprocess_data` first.")

    data_module = ImdbReviewsDataModule(
        train_path=train_path,
        test_path=test_path,
        text_field=data_cfg.preprocess.text_field,
        label_field=data_cfg.preprocess.label_field,
        split=SplitConfig(
            val_size=float(split_cfg.val_size),
            random_state=int(split_cfg.random_state),
            shuffle=bool(split_cfg.shuffle),
        ),
        pretrained_model_name=model_cfg.pretrained_model_name,
        max_length=int(model_cfg.max_length),
        train_batch_size=int(train_cfg.bert_trainer.batch_size),
        eval_batch_size=int(train_cfg.bert_trainer.batch_size),
        num_workers=int(train_cfg.bert_trainer.num_workers),
    )

    mlflow.set_tracking_uri(logging_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(logging_cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="bert") as run:
        commit = git.Repo(str(ROOT_DIR)).head.commit.hexsha
        mlf_logger = MLFlowLogger(
            tracking_uri=logging_cfg.mlflow.tracking_uri,
            run_id=run.info.run_id,
            log_model=True,
            tags={"git_commit": commit},
        )
        mlf_logger.log_hyperparams(
            {
                "model_type": "bert",
                "pretrained_model_name": model_cfg.pretrained_model_name,
                "max_length": int(model_cfg.max_length),
                "lr": float(train_cfg.optimizer.lr),
                "weight_decay": float(train_cfg.optimizer.weight_decay),
                "batch_size": int(train_cfg.bert_trainer.batch_size),
                "max_epochs": int(train_cfg.bert_trainer.max_epochs),
                "val_size": float(split_cfg.val_size),
                "random_state": int(split_cfg.random_state),
                "git_commit": commit,
            }
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=ROOT_DIR / train_cfg.bert_trainer.checkpoints_dir,
            monitor="val/f1_neg",
            mode="max",
            save_top_k=1,
            filename="bert-{epoch:02d}",
        )
        model = BertSentimentModule(cfg)
        trainer = L.Trainer(
            max_epochs=int(train_cfg.bert_trainer.max_epochs),
            logger=mlf_logger,
            callbacks=[ckpt_cb],
            log_every_n_steps=logging_cfg.log_every_n_steps,
            accelerator=train_cfg.bert_trainer.accelerator,
            devices=train_cfg.bert_trainer.devices,
            precision=train_cfg.bert_trainer.precision,
        )
        trainer.fit(model, datamodule=data_module)

        best_ckpt_path = ckpt_cb.best_model_path
        best_model = BertSentimentModule.load_from_checkpoint(best_ckpt_path, cfg=cfg)
        best_model.eval()

        trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt_path)

        mlflow.pytorch.log_model(best_model, name="bert_with_head_model")


def train_bert(overrides: list[str] | None = None) -> None:
    overrides = list(overrides or [])
    if "model=bert" not in overrides:
        overrides.append("model=bert")

    cfg = get_cfg(overrides)
    _train(cfg)


def main() -> None:
    fire.Fire(train_bert)


if __name__ == "__main__":
    main()
