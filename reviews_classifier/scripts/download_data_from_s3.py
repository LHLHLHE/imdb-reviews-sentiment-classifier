from pathlib import Path

import fire
from dvc.repo import Repo
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[2]
HYDRA_CONFIG_DIR = ROOT_DIR / "configs"


def get_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def download(overrides: list[str] | None = None) -> None:
    cfg = get_cfg(overrides)
    repo = Repo(str(ROOT_DIR))
    repo.pull(targets=[cfg.data.train_data_path, cfg.data.test_data_path], allow_missing=False)
    print(f"Downloaded data: {cfg.data.train_data_path}, {cfg.data.test_data_path}")


def main() -> None:
    fire.Fire(download)


if __name__ == "__main__":
    main()
