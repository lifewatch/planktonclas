from pathlib import Path
from tempfile import TemporaryDirectory

from planktonclas import config, paths


def main():
    original_conf_path = config.CONF_PATH
    original_paths_conf = paths.CONF

    try:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            previous_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp_path)
                config.set_config_path(config.DEFAULT_CONFIG_PATH)
                paths.CONF = None

                assert config.CONFIG_ROOT == str(tmp_path)
                assert paths.get_base_dir() == str(tmp_path)
                assert paths.get_images_dir() == str(tmp_path / "data" / "images")
            finally:
                os.chdir(previous_cwd)

        assert config.normalize_user_path(r".\data/images", path_separator="/") == "data/images"
        assert config.normalize_user_path("./data\\images", path_separator="/") == "data/images"
        assert config.normalize_user_path("./data/images", path_separator="\\") == r"data\images"
    finally:
        config.set_config_path(original_conf_path)
        paths.CONF = original_paths_conf

    print("Path sanity checks passed.")


if __name__ == "__main__":
    main()
