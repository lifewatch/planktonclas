"""
Fast compatibility smoke runner for planktonclas.

This script does two related jobs:
1. checks whether configured model names are still available in the current
   TensorFlow/Keras environment
2. runs a small matrix of real project workflows against short training runs

It is intentionally a smoke test, not an exhaustive benchmark. The goal is to
catch broken configs, model-loading regressions, and workflow issues quickly.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml
from tensorflow.keras import applications

from planktonclas import cli


STANDARD_MODEL_NAMES = [
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "InceptionResNetV2",
    "InceptionV3",
    "MobileNet",
    "NASNetMobile",
    "Xception",
    "ResNet50",
    "VGG16",
    "VGG19",
    "EfficientNetV2B0",
]
LOCAL_PRETRAINED_MODEL_NAMES = ["Phytoplankton_EfficientNetV2B0"]


@dataclass
class SmokeCase:
    name: str
    modelname: str
    mode: str
    use_validation: bool
    use_test: bool
    use_tensorboard: bool = False


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _supports_tensorboard_autolaunch() -> bool:
    return shutil.which("tensorboard") is not None


def _available_model_status(project_models_dir: Path) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    for name in STANDARD_MODEL_NAMES:
        ok = hasattr(applications, name)
        rows.append((name, ok, "tf.keras.applications"))
    for name in LOCAL_PRETRAINED_MODEL_NAMES:
        ok = (project_models_dir / name).is_dir()
        rows.append((name, ok, "local pretrained model directory"))
    return rows


def _build_cases(models: list[str], include_tensorboard: bool) -> list[SmokeCase]:
    cases: list[SmokeCase] = []
    base_shapes = [
        ("fast_minimal", "fast", False, False, False),
        ("fast_val_test", "fast", True, True, False),
        ("normal_val_no_test", "normal", True, False, False),
    ]
    if include_tensorboard:
        base_shapes.append(("fast_tensorboard", "fast", True, False, True))

    for modelname in models:
        for name, mode, use_validation, use_test, use_tensorboard in base_shapes:
            cases.append(
                SmokeCase(
                    name=f"{modelname}__{name}",
                    modelname=modelname,
                    mode=mode,
                    use_validation=use_validation,
                    use_test=use_test,
                    use_tensorboard=use_tensorboard,
                )
            )
    return cases


def _prepare_case_config(config_path: Path, case: SmokeCase, epochs: int, batch_size: int, image_size: int) -> None:
    conf = _load_yaml(config_path)
    conf = copy.deepcopy(conf)

    conf["model"]["modelname"]["value"] = case.modelname
    conf["model"]["image_size"]["value"] = image_size

    conf["training"]["mode"]["value"] = case.mode
    conf["training"]["epochs"]["value"] = epochs
    conf["training"]["batch_size"]["value"] = batch_size
    conf["training"]["use_validation"]["value"] = case.use_validation
    conf["training"]["use_test"]["value"] = case.use_test
    conf["training"]["use_early_stopping"]["value"] = False

    conf["monitor"]["use_tensorboard"]["value"] = case.use_tensorboard

    conf["augmentation"]["use_augmentation"]["value"] = False
    conf["dataset"]["split_ratios"]["value"] = [0.8, 0.1, 0.1]

    _dump_yaml(config_path, conf)


def _run_cli(argv: list[str]) -> None:
    cli.main(argv)


def _run_case(project_dir: Path, case: SmokeCase, epochs: int, batch_size: int, image_size: int, with_report: bool) -> None:
    config_path = project_dir / "config.yaml"
    _prepare_case_config(config_path, case, epochs=epochs, batch_size=batch_size, image_size=image_size)

    _run_cli(["validate-config", str(project_dir)])
    _run_cli(["train", str(project_dir), "--workers", "1"])

    if with_report and case.use_test:
        _run_cli(["report", str(project_dir), "--mode", "quick"])


def _print_model_status(rows: list[tuple[str, bool, str]]) -> None:
    print("\nModel availability")
    print("=" * 72)
    for name, ok, source in rows:
        status = "OK" if ok else "MISSING"
        print(f"{name:<32} {status:<8} {source}")


def _recommendations() -> None:
    print("\nModel guidance")
    print("=" * 72)
    print("EfficientNetV2B0 is still the strongest default in this package.")
    print("DenseNet/ResNet/MobileNet/Xception/Inception families are older, but they")
    print("are not obsolete just because they are older. Keep them if you want:")
    print("- backwards compatibility with older projects")
    print("- side-by-side benchmarking on your own data")
    print("- smaller/faster alternatives such as MobileNet")
    print("VGG16/VGG19 are the oldest and least attractive defaults today, but they can")
    print("still be useful as baseline architectures.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fast config/model smoke matrix.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["EfficientNetV2B0"],
        help="Model names to train in the smoke matrix. Default: EfficientNetV2B0",
    )
    parser.add_argument(
        "--all-standard-models",
        action="store_true",
        help="Train the smoke matrix for all standard tf.keras application models in the config.",
    )
    parser.add_argument(
        "--include-local-pretrained",
        action="store_true",
        help="Also include Phytoplankton_EfficientNetV2B0 if it is available under models/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs per smoke case. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per smoke case. Default: 2",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=100,
        help="Image size for smoke runs. Default: 100",
    )
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="Also run `planktonclas report` for cases that keep a test split.",
    )
    parser.add_argument(
        "--with-tensorboard-case",
        action="store_true",
        help="Include one TensorBoard-enabled smoke case per model when the tensorboard executable is available.",
    )
    parser.add_argument(
        "--keep-projects",
        action="store_true",
        help="Keep temporary smoke projects on disk for inspection.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    project_models_dir = repo_root / "models"

    model_rows = _available_model_status(project_models_dir)
    _print_model_status(model_rows)
    _recommendations()

    selected_models = list(args.models)
    if args.all_standard_models:
        selected_models = list(STANDARD_MODEL_NAMES)
    if args.include_local_pretrained and (project_models_dir / "Phytoplankton_EfficientNetV2B0").is_dir():
        if "Phytoplankton_EfficientNetV2B0" not in selected_models:
            selected_models.append("Phytoplankton_EfficientNetV2B0")

    if not selected_models:
        print("\nNo models selected for smoke training.")
        return 0

    include_tensorboard = args.with_tensorboard_case and _supports_tensorboard_autolaunch()
    if args.with_tensorboard_case and not include_tensorboard:
        print("\nSkipping TensorBoard smoke cases because the `tensorboard` executable is not on PATH.")

    cases = _build_cases(selected_models, include_tensorboard=include_tensorboard)
    print(f"\nPrepared {len(cases)} smoke case(s).")

    failures: list[tuple[str, str]] = []

    for index, case in enumerate(cases, start=1):
        print("\n" + "=" * 72)
        print(f"[{index}/{len(cases)}] Running {case.name}")
        print("=" * 72)

        tmp_path = Path(tempfile.mkdtemp(prefix="planktonclas-smoke-"))
        project_dir = tmp_path / case.name

        try:
            _run_cli(["init", str(project_dir), "--demo"])
            _run_cli(["notebooks", str(project_dir)])
            _run_case(
                project_dir=project_dir,
                case=case,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                with_report=args.with_report,
            )
            print(f"PASS {case.name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((case.name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {case.name}")
            traceback.print_exc()
            if args.keep_projects:
                print(f"Kept failing project at: {project_dir}")
        finally:
            if args.keep_projects:
                print(f"Kept project at: {project_dir}")
            else:
                shutil.rmtree(tmp_path, ignore_errors=True)

    print("\nSummary")
    print("=" * 72)
    print(f"Cases run: {len(cases)}")
    print(f"Failures: {len(failures)}")
    if failures:
        for name, message in failures:
            print(f"- {name}: {message}")
        return 1

    print("All smoke cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
