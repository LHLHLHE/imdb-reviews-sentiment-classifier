import fire

from reviews_classifier.scripts.download_data_from_s3 import download


def main() -> None:
    fire.Fire(
        {
            "download_data_from_s3": download,
        }
    )


if __name__ == "__main__":
    main()
