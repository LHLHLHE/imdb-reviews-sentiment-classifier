import json
import re
from html import unescape
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from reviews_classifier.utils import check_data_exists

ROOT_DIR = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = ROOT_DIR / "configs"

HTML_RE = re.compile(r"<[^>]+>")


def get_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def _clean_text(text: str, preprocess_cfg: DictConfig) -> str:
    if preprocess_cfg.strip_html:
        text = unescape(HTML_RE.sub(" ", text))
    if preprocess_cfg.normalize_whitespace:
        text = " ".join(text.split())
    if preprocess_cfg.lowercase:
        text = text.lower()
    return text


def _preprocess_file(in_path: Path, out_path: Path, preprocess_cfg: DictConfig) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text_key = preprocess_cfg.text_field
    label_key = preprocess_cfg.label_field
    min_chars = int(preprocess_cfg.min_chars)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)

            text = str(obj[text_key])
            label = int(obj[label_key])

            text = _clean_text(text, preprocess_cfg)
            if len(text) < min_chars:
                continue

            fout.write(json.dumps({text_key: text, label_key: label}, ensure_ascii=False) + "\n")


def preprocess(overrides: list[str] | None = None) -> None:
    cfg = get_cfg(overrides)
    data_cfg = cfg.data
    preprocess_cfg = data_cfg.preprocess

    train_raw_path = ROOT_DIR / data_cfg.train_data_path
    test_raw_path = ROOT_DIR / data_cfg.test_data_path

    data_exists, missing_str = check_data_exists(train_raw_path, test_raw_path)
    if not data_exists:
        raise SystemExit(f"Raw data not found: {missing_str}. Run `download_data_from_s3` first.")

    train_processed_path = ROOT_DIR / preprocess_cfg.processed_train_data_path
    test_processed_path = ROOT_DIR / preprocess_cfg.processed_test_data_path

    _preprocess_file(train_raw_path, train_processed_path, preprocess_cfg)
    _preprocess_file(test_raw_path, test_processed_path, preprocess_cfg)

    print(f"Preprocessed -> {train_processed_path}, {test_processed_path}")


def main() -> None:
    fire.Fire(preprocess)


if __name__ == "__main__":
    main()
