import sys
from pathlib import Path
from types import ModuleType

from planktonclas import cli


def test_cli_init_creates_basic_project_structure(tmp_path):
    project_dir = tmp_path / "my_project"

    cli.main(["init", str(project_dir)])

    assert (project_dir / "config.yaml").is_file()
    assert (project_dir / "data" / "images").is_dir()
    assert (project_dir / "data" / "dataset_files").is_dir()
    assert (project_dir / "models").is_dir()


def test_cli_validate_config_accepts_project_directory(tmp_path, capsys):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])

    cli.main(["validate-config", str(project_dir)])

    output = capsys.readouterr().out
    assert f"Configuration OK: {project_dir / 'config.yaml'}" in output
    assert "Images directory:" in output
    assert "Models directory:" in output


def test_cli_train_accepts_project_directory(tmp_path, monkeypatch):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])

    captured = {}
    fake_module = ModuleType("planktonclas.train_runfile")

    def fake_train_fn(TIMESTAMP, CONF):
        captured["timestamp"] = TIMESTAMP
        captured["conf"] = CONF

    fake_module.train_fn = fake_train_fn
    monkeypatch.setitem(sys.modules, "planktonclas.train_runfile", fake_module)

    cli.main(["train", str(project_dir), "--quick", "--workers", "1"])

    assert "timestamp" in captured
    assert captured["conf"]["dataset"]["num_workers"] == 1
    assert captured["conf"]["training"]["mode"] == "fast"
    assert captured["conf"]["training"]["epochs"] == 1


def test_cli_pretrained_uses_shared_pretrained_helper(tmp_path, monkeypatch, capsys):
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    captured = {}

    def fake_ensure_pretrained_model(models_dir, modelname, version="latest", force=False):
        captured["models_dir"] = models_dir
        captured["modelname"] = modelname
        captured["version"] = version
        captured["force"] = force
        return str(Path(models_dir) / modelname)

    monkeypatch.setattr(cli.model_utils, "ensure_pretrained_model", fake_ensure_pretrained_model)
    monkeypatch.setattr(
        cli.model_utils,
        "get_pretrained_metadata",
        lambda name, version="latest": {
            "name": name,
            "architecture": "EfficientNetV2B0",
            "version": version,
            "checkpoint_name": "best_model.keras",
        },
    )

    cli.main(["pretrained", str(project_dir), "--model", "FlowCyto", "--version", "latest"])

    output = capsys.readouterr().out
    assert "Pretrained model available at:" in output
    assert "Architecture: EfficientNetV2B0" in output
    assert "Version: latest" in output
    assert "Checkpoint: best_model.keras" in output
    assert captured["models_dir"] == str(project_dir / "models")
    assert captured["modelname"] == "FlowCyto"
    assert captured["version"] == "latest"
    assert captured["force"] is False


def test_cli_list_models_shows_published_pretrained_metadata(tmp_path, monkeypatch, capsys):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])
    (project_dir / "models" / "FlowCam").mkdir()
    (project_dir / "models" / "custom_run").mkdir()

    monkeypatch.setattr(
        cli.model_utils,
        "get_pretrained_metadata",
        lambda name, version="latest": {
            "name": name,
            "architecture": "EfficientNetV2B0",
            "version": version,
            "checkpoint_name": "final_model.h5",
        },
    )

    cli.main(["list-models", str(project_dir)])

    output = capsys.readouterr().out
    assert "FlowCam | architecture=EfficientNetV2B0 | version=latest | checkpoint=final_model.h5" in output
    assert "custom_run" in output
