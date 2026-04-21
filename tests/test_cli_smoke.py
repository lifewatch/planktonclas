import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from planktonclas import cli


def test_cli_init_creates_basic_project_structure(tmp_path):
    project_dir = tmp_path / "my_project"

    cli.main(["init", str(project_dir)])

    assert (project_dir / "config.yaml").is_file()
    assert (project_dir / "data" / "images").is_dir()
    assert (project_dir / "data" / "dataset_files").is_dir()
    assert (project_dir / "models").is_dir()


def test_cli_validate_config_runs_on_initialized_project(tmp_path, capsys):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])

    cli.main(["validate-config", "--config", str(project_dir / "config.yaml")])

    output = capsys.readouterr().out
    assert "Configuration OK:" in output
    assert "Images directory:" in output
    assert "Models directory:" in output


def test_cli_train_quick_invokes_training_entrypoint(tmp_path, monkeypatch):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])

    captured = {}
    fake_module = ModuleType("planktonclas.train_runfile")

    def fake_train_fn(TIMESTAMP, CONF):
        captured["timestamp"] = TIMESTAMP
        captured["conf"] = CONF

    fake_module.train_fn = fake_train_fn
    monkeypatch.setitem(sys.modules, "planktonclas.train_runfile", fake_module)

    cli.main(
        [
            "train",
            "--config",
            str(project_dir / "config.yaml"),
            "--quick",
            "--workers",
            "1",
        ]
    )

    assert "timestamp" in captured
    assert captured["conf"]["dataset"]["num_workers"] == 1
    assert captured["conf"]["training"]["mode"] == "fast"
    assert captured["conf"]["training"]["epochs"] == 1


def test_cli_api_command_starts_deepaas_runner(tmp_path, monkeypatch):
    project_dir = tmp_path / "my_project"
    cli.main(["init", str(project_dir)])

    captured = {}

    def fake_resolve_executable(name):
        assert name == "deepaas-run"
        return "deepaas-run"

    def fake_subprocess_run(command, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "_resolve_executable", fake_resolve_executable)
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "api",
                "--config",
                str(project_dir / "config.yaml"),
                "--host",
                "127.0.0.1",
                "--port",
                "5001",
            ]
        )

    assert exc_info.value.code == 0
    assert captured["command"] == [
        "deepaas-run",
        "--listen-ip",
        "127.0.0.1",
        "--listen-port",
        "5001",
    ]
    assert captured["env"]["PLANKTONCLAS_CONFIG"] == str(project_dir / "config.yaml")
    assert captured["env"]["DEEPAAS_V2_MODEL"] == "planktonclas"


def test_cli_docker_command_builds_inference_image(tmp_path, monkeypatch, capsys):
    project_dir = tmp_path / "my_project"
    model_dir = project_dir / "models" / "2026-04-21_120000" / "ckpts"
    model_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "config.yaml").write_text("general:\n  base_directory:\n    value: .\n", encoding="utf-8")
    (model_dir / "best_model.keras").write_text("stub", encoding="utf-8")

    captured = {}

    def fake_resolve_executable(name):
        assert name == "docker"
        return "docker"

    def fake_copy_docker_build_context(context_dir, project_dir_value, timestamp, ckpt_name):
        captured["context_dir"] = context_dir
        captured["project_dir"] = project_dir_value
        captured["timestamp"] = timestamp
        captured["ckpt_name"] = ckpt_name

    def fake_subprocess_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "_resolve_executable", fake_resolve_executable)
    monkeypatch.setattr(cli, "_copy_docker_build_context", fake_copy_docker_build_context)
    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)

    cli.main(["docker", str(project_dir), "--tag", "my-plankton-api:latest"])

    output = capsys.readouterr().out
    assert "Docker image built successfully: my-plankton-api:latest" in output
    assert captured["project_dir"] == str(project_dir.resolve())
    assert captured["timestamp"] == "2026-04-21_120000"
    assert captured["ckpt_name"] == "best_model.keras"
    assert captured["command"][0] == "docker"
    assert captured["command"][1] == "build"
    assert "--tag" in captured["command"]
    assert "my-plankton-api:latest" in captured["command"]
