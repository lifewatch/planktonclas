from pathlib import Path

from planktonclass.data_utils import (
    create_data_splits,
    load_class_names,
    load_data_splits,
    split_file_has_entries,
)


def test_dataset_split_generation_and_loading_smoke(tmp_path):
    images_dir = tmp_path / "images"
    splits_dir = tmp_path / "dataset_files"

    for class_name, filenames in {
        "ClassA": ["a1.jpg", "a2.jpg"],
        "ClassB": ["b1.jpg"],
    }.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (class_dir / filename).write_bytes(b"jpg")

    create_data_splits(str(splits_dir), str(images_dir), split_ratios=[1, 0, 0])

    assert split_file_has_entries(str(splits_dir), "train")
    assert load_class_names(str(splits_dir)).tolist() == ["ClassA", "ClassB"]

    X_train, y_train = load_data_splits(str(splits_dir), str(images_dir), "train")
    relative_paths = sorted(
        str(Path(path).relative_to(images_dir)).replace("\\", "/") for path in X_train
    )
    assert relative_paths == ["ClassA/a1.jpg", "ClassA/a2.jpg", "ClassB/b1.jpg"]
    assert y_train.tolist() == [0, 0, 1]
