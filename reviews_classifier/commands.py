import fire

from reviews_classifier.scripts.download_data_from_s3 import download
from reviews_classifier.scripts.preprocess_data import preprocess
from reviews_classifier.scripts.train import train_baseline, train_bert


def main() -> None:
    fire.Fire(
        {
            "download_data_from_s3": download,
            "preprocess_data": preprocess,
            "train": {
                "baseline": train_baseline,
                "bert": train_bert,
            },
        }
    )


if __name__ == "__main__":
    main()
