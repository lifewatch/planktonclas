import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planktonclass import config, paths


def _normalized_path(path):
    return Path(path).resolve().as_posix().lower()


def main():
    original_conf_path = config.CONF_PATH
    original_paths_conf = paths.CONF
    synthetic_unc_path = r"\\example-server\example-share\plankton-images"

    try:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            previous_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmp_path)
                config.set_config_path(config.DEFAULT_CONFIG_PATH)
                paths.CONF = None

                assert _normalized_path(config.CONFIG_ROOT) == _normalized_path(tmp_path)
                assert _normalized_path(paths.get_base_dir()) == _normalized_path(tmp_path)
                assert _normalized_path(paths.get_images_dir()) == _normalized_path(tmp_path / "data" / "images")
            finally:
                os.chdir(previous_cwd)

        assert config.normalize_user_path(r".\data/images", path_separator="/") == "data/images"
        assert config.normalize_user_path("./data\\images", path_separator="/") == "data/images"
        assert config.normalize_user_path("./data/images", path_separator="\\") == r"data\images"

        raw_conf = """
general:
  base_directory:
    value: "."
  images_directory:
    value: "\\\\example-server\\example-share\\plankton-images"
testing:
  output_directory:
    value: null
augmentation:
  use_augmentation:
    value: false
"""
        conf = config.load_yaml_config(raw_conf)
        conf_dict = config.get_conf_dict(conf)
        assert conf_dict["general"]["images_directory"] == synthetic_unc_path

        raw_conf_forward = """
general:
  base_directory:
    value: "."
  images_directory:
    value: "//example-server/example-share/plankton-images"
testing:
  output_directory:
    value: null
augmentation:
  use_augmentation:
    value: false
"""
        conf_forward = config.load_yaml_config(raw_conf_forward)
        conf_forward_dict = config.get_conf_dict(conf_forward)
        assert conf_forward_dict["general"]["images_directory"] == synthetic_unc_path
    finally:
        config.set_config_path(original_conf_path)
        paths.CONF = original_paths_conf

    print("Path sanity checks passed.")


if __name__ == "__main__":
    main()
