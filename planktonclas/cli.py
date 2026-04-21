"""
Command-line interface for planktonclas.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from importlib.resources import files

import requests

from planktonclas import config, paths


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROJECT_CONFIG_NAME = "config.yaml"


def _resource_path(*parts):
    return os.fspath(files("planktonclas").joinpath(*parts))


DEFAULT_NOTEBOOKS_DIR = _resource_path("resources", "notebooks")
DEFAULT_DEMO_IMAGES_DIR = _resource_path("resources", "demo-images")
DEFAULT_DEMO_SPLITS_DIR = _resource_path("resources", "dataset_files")
DEFAULT_TRANSFORMATION_DATA_DIR = _resource_path("resources", "data_transformation")
PRETRAINED_MODEL_NAME = "Phytoplankton_EfficientNetV2B0"
PRETRAINED_MODEL_TAR = f"{PRETRAINED_MODEL_NAME}.tar.gz"
PRETRAINED_MODEL_URL = (
    f"https://zenodo.org/records/15269453/files/{PRETRAINED_MODEL_TAR}?download=1"
)


def _default_config_path():
    cwd_config = os.path.abspath(DEFAULT_PROJECT_CONFIG_NAME)
    if os.path.exists(cwd_config):
        return cwd_config
    return config.DEFAULT_CONFIG_PATH


def _apply_config(conf_path):
    config.set_config_path(conf_path)
    paths.CONF = config.get_conf_dict()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_placeholder(path, contents=""):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(contents)


def _copy_tree(src, dst):
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Missing resource directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _resolve_executable(name):
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Executable not found in PATH: {name}")
    return executable


def _resolve_project_dir(directory=None, conf_path=None):
    if conf_path:
        _apply_config(os.path.abspath(conf_path))
        return paths.get_base_dir()
    if directory is not None:
        return os.path.abspath(directory)
    return os.path.abspath(".")


def _safe_extract_tar(archive_path, destination):
    destination = os.path.abspath(destination)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, member_path]) != destination:
                raise ValueError(f"Unsafe path found in tar archive: {member.name}")
            if member.isdir():
                os.makedirs(member_path, exist_ok=True)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            os.makedirs(os.path.dirname(member_path), exist_ok=True)
            with extracted, open(member_path, "wb") as dst:
                shutil.copyfileobj(extracted, dst)


def _display_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path


def _list_model_timestamps():
    models_dir = paths.get_models_dir()
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"No models directory found: {models_dir}")

    timestamps = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
        ]
    )
    if not timestamps:
        raise FileNotFoundError(f"No models found in: {models_dir}")
    return timestamps


def _choose_report_timestamp(explicit_timestamp=None):
    if explicit_timestamp:
        return explicit_timestamp

    timestamps = _list_model_timestamps()
    suggested = timestamps[-1]

    if len(timestamps) == 1:
        print(f"Only one model run found. Using: {suggested}")
        return suggested

    print("Available model runs:")
    for idx, timestamp in enumerate(timestamps, start=1):
        marker = " (suggested)" if timestamp == suggested else ""
        print(f"  {idx}. {timestamp}{marker}")

    print(f"Suggested most recent run: {suggested}")
    print("Press Enter to use the suggested run, or type a number from the list.")

    while True:
        selection = input("Report model selection: ").strip()
        if not selection:
            return suggested
        if selection.isdigit():
            choice = int(selection)
            if 1 <= choice <= len(timestamps):
                return timestamps[choice - 1]
        print(f"Please enter a number between 1 and {len(timestamps)}, or press Enter.")


def _choose_report_mode(explicit_mode=None):
    if explicit_mode:
        return explicit_mode

    print("Report detail level:")
    print("  1. quick (suggested) - core figures only")
    print("  2. full - also generates the threshold-based figures in results subfolders")
    print("Press Enter to use quick, or type 1 or 2.")

    while True:
        selection = input("Report mode selection: ").strip().lower()
        if selection in {"", "1", "quick"}:
            return "quick"
        if selection in {"2", "full"}:
            return "full"
        print("Please enter 1, 2, quick, full, or press Enter.")


def init_project(args):
    target_dir = os.path.abspath(args.directory)
    config_path = os.path.join(target_dir, DEFAULT_PROJECT_CONFIG_NAME)

    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(
            f"{config_path} already exists. Use --force to overwrite it."
        )

    _ensure_dir(target_dir)
    _ensure_dir(os.path.join(target_dir, "data", "images"))
    _ensure_dir(os.path.join(target_dir, "data", "dataset_files"))
    _ensure_dir(os.path.join(target_dir, "models"))

    shutil.copyfile(config.DEFAULT_CONFIG_PATH, config_path)

    if args.demo:
        _copy_tree(DEFAULT_DEMO_IMAGES_DIR, os.path.join(target_dir, "data", "images"))
        _copy_tree(
            DEFAULT_DEMO_SPLITS_DIR,
            os.path.join(target_dir, "data", "dataset_files"),
        )
    else:
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "classes.txt"),
            "# one class name per line\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "train.txt"),
            "# relative/image/path.jpg\t0\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "val.txt"),
            "# relative/image/path.jpg\t0\n",
        )
        _write_placeholder(
            os.path.join(target_dir, "data", "dataset_files", "test.txt"),
            "# relative/image/path.jpg\t0\n",
        )

    print(f"Initialized project at: {target_dir}")
    print(f"Config: {config_path}")
    print(f"Images: {os.path.join(target_dir, 'data', 'images')}")
    print(f"Dataset files: {os.path.join(target_dir, 'data', 'dataset_files')}")
    print(f"Models: {os.path.join(target_dir, 'models')}")
    if args.demo:
        print("Demo data copied into data/images and data/dataset_files.")


def validate_config(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)
    print(f"Configuration OK: {config.CONF_PATH}")
    print(f"Base directory: {paths.get_base_dir()}")
    print(f"Images directory: {paths.get_images_dir()}")
    print(f"Splits directory: {paths.get_splits_dir()}")
    print(f"Models directory: {paths.get_models_dir()}")


def train_model(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    from planktonclas.train_runfile import train_fn

    conf = config.get_conf_dict()
    conf["dataset"]["num_workers"] = args.workers
    if args.mode:
        conf["training"]["mode"] = args.mode
    if args.epochs is not None:
        conf["training"]["epochs"] = args.epochs
    if args.quick:
        conf["training"]["mode"] = "fast"
        conf["training"]["epochs"] = 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    train_fn(TIMESTAMP=timestamp, CONF=conf)


def generate_report_cmd(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    from planktonclas.report_utils import generate_report

    selected_timestamp = _choose_report_timestamp(args.timestamp)
    selected_mode = _choose_report_mode(args.mode)
    print("Starting report generation...")
    summary = generate_report(
        timestamp=selected_timestamp,
        mode=selected_mode,
        progress=lambda message: print(f"[report] {message}"),
    )
    print(f"Report generated for timestamp: {summary['timestamp']}")
    print(f"Mode: {summary['mode']}")
    print(f"Results: {_display_path(summary['results_dir'])}")
    print(f"Predictions: {_display_path(summary['predictions_file'])}")
    print(f"Top-1 accuracy: {summary['top1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {summary['top3_accuracy']:.3f}")
    print(f"Top-5 accuracy: {summary['top5_accuracy']:.3f}")
    print(f"Macro F1: {summary['macro_f1']:.3f}")
    print(f"Weighted F1: {summary['weighted_f1']:.3f}")


def run_api(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    env = os.environ.copy()
    env[config.CONFIG_ENV_VAR] = conf_path
    env["DEEPAAS_V2_MODEL"] = "planktonclas"

    command = [_resolve_executable("deepaas-run"), "--listen-ip", args.host]
    if args.port is not None:
        command.extend(["--listen-port", str(args.port)])

    completed = subprocess.run(command, env=env, check=False)
    raise SystemExit(completed.returncode)


def list_models(args):
    conf_path = os.path.abspath(args.config or _default_config_path())
    _apply_config(conf_path)

    models_dir = paths.get_models_dir()
    if not os.path.isdir(models_dir):
        print(f"No models directory found: {models_dir}")
        return

    entries = sorted(
        [
            name
            for name in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, name))
        ]
    )
    if not entries:
        print(f"No models found in: {models_dir}")
        return

    print(f"Models in {models_dir}:")
    for name in entries:
        print(name)


def notebooks(args):
    project_dir = _resolve_project_dir(args.directory, args.config)
    target_dir = os.path.join(project_dir, "notebooks")
    transformation_dir = os.path.join(project_dir, "data", "data_transformation")
    _ensure_dir(project_dir)
    _ensure_dir(os.path.join(project_dir, "data"))

    if os.path.exists(target_dir) and not args.force:
        raise FileExistsError(
            f"{target_dir} already exists. Use --force to overwrite notebook files."
        )

    if os.path.exists(transformation_dir) and not args.force:
        print(
            f"Transformation data directory already exists: {transformation_dir}"
        )
        print("Use --force to refresh the packaged transformation data files.")

    _copy_tree(DEFAULT_NOTEBOOKS_DIR, target_dir)
    _copy_tree(DEFAULT_TRANSFORMATION_DATA_DIR, transformation_dir)
    print(f"Notebooks copied to: {target_dir}")
    print(f"Transformation data copied to: {transformation_dir}")


def download_pretrained(args):
    project_dir = _resolve_project_dir(args.directory, args.config)
    models_dir = os.path.join(project_dir, "models")
    target_dir = os.path.join(models_dir, PRETRAINED_MODEL_NAME)
    _ensure_dir(models_dir)

    if os.path.exists(target_dir) and not args.force:
        raise FileExistsError(
            f"{target_dir} already exists. Use --force to re-download the pretrained model."
        )

    print(f"Downloading pretrained model: {PRETRAINED_MODEL_NAME}")
    print(f"Source: {PRETRAINED_MODEL_URL}")

    temp_archive = None
    try:
        with requests.get(PRETRAINED_MODEL_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", delete=False
            ) as tmp_file:
                temp_archive = tmp_file.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_file.write(chunk)

        _safe_extract_tar(temp_archive, models_dir)
    finally:
        if temp_archive and os.path.exists(temp_archive):
            os.remove(temp_archive)

    print(f"Pretrained model available at: {target_dir}")


def build_parser():
    parser = argparse.ArgumentParser(prog="planktonclas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a local planktonclas project structure."
    )
    init_parser.add_argument("directory", nargs="?", default=".")
    init_parser.add_argument(
        "--demo",
        action="store_true",
        help="Populate the project with demo images and demo dataset files.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yaml in the target directory.",
    )
    init_parser.set_defaults(func=init_project)

    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a config file and print resolved paths."
    )
    validate_parser.add_argument("--config")
    validate_parser.set_defaults(func=validate_config)

    train_parser = subparsers.add_parser(
        "train", help="Train a model using a config file."
    )
    train_parser.add_argument("--config")
    train_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataset preprocessing workers.",
    )
    train_parser.add_argument(
        "--mode",
        choices=["normal", "fast"],
        help="Override the training mode from the config file.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override the number of training epochs from the config file.",
    )
    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test run: uses fast mode and 1 epoch.",
    )
    train_parser.set_defaults(func=train_model)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate evaluation plots and metrics for a trained run.",
    )
    report_parser.add_argument("--config")
    report_parser.add_argument(
        "--timestamp",
        help="Timestamped model directory to report on. Defaults to the latest run.",
    )
    report_parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        help="Report detail level. Quick skips the subfolder threshold plots.",
    )
    report_parser.set_defaults(func=generate_report_cmd)

    api_parser = subparsers.add_parser(
        "api", help="Launch the DEEPaaS API with a selected config file."
    )
    api_parser.add_argument("--config")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=5000)
    api_parser.set_defaults(func=run_api)

    models_parser = subparsers.add_parser(
        "list-models", help="List models inside the configured models directory."
    )
    models_parser.add_argument("--config")
    models_parser.set_defaults(func=list_models)

    notebooks_parser = subparsers.add_parser(
        "notebooks", help="Copy packaged notebooks into a project directory."
    )
    notebooks_parser.add_argument("directory", nargs="?", default=".")
    notebooks_parser.add_argument("--config")
    notebooks_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite notebook files in an existing notebooks directory.",
    )
    notebooks_parser.set_defaults(func=notebooks)

    pretrained_parser = subparsers.add_parser(
        "pretrained",
        help="Download the packaged pretrained phytoplankton model into a project.",
    )
    pretrained_parser.add_argument("directory", nargs="?", default=".")
    pretrained_parser.add_argument("--config")
    pretrained_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download into an existing pretrained model directory.",
    )
    pretrained_parser.set_defaults(func=download_pretrained)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
