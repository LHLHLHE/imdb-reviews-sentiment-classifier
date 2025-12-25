def check_data_exists(train_path, test_path):
    missing = [path for path in (train_path, test_path) if not path.exists()]
    if missing:
        return False, ", ".join(str(path) for path in missing)
    return True, ""
