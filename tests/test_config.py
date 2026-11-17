"""Config loading: extends, deep merge and ${output_dir} expansion."""

from medqa.config import load_config, merge


def test_merge_is_recursive_and_non_destructive() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merge(base, {"a": {"y": 20}, "c": 4})
    assert out == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}
    assert base["a"]["y"] == 2


def test_debug_extends_base_and_expands_output_dir() -> None:
    base = load_config("configs/base.yaml")
    debug = load_config("configs/debug.yaml")
    assert debug["data"]["train_size"] == 50
    assert debug["model"]["name"] == base["model"]["name"]
    assert debug["wandb"]["enabled"] is False
    assert debug["data"]["splits_dir"] == f"{debug['output_dir']}/splits"
    assert "${output_dir}" not in str(debug)


def test_child_can_override_output_dir(tmp_path) -> None:
    (tmp_path / "parent.yaml").write_text(
        "output_dir: a\ndata:\n  splits_dir: ${output_dir}/splits\n"
    )
    (tmp_path / "child.yaml").write_text("extends: parent.yaml\noutput_dir: b\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["data"]["splits_dir"] == "b/splits"
