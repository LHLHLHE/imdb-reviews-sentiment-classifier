def check_data_exists(train_path, test_path):
    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        return False, ", ".join(str(p) for p in missing)
    return True, ""
