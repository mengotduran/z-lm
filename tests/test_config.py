from pathlib import Path

import pytest

from zlm.config import Config, config_from_dict, load_config

CONFIGS = sorted(Path(__file__).resolve().parent.parent.glob("configs/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.log.run_name == path.stem


def test_defaults_when_empty():
    assert config_from_dict({}) == Config()


def test_values_override_defaults():
    cfg = config_from_dict({"model": {"n_layer": 6}})
    assert cfg.model.n_layer == 6
    assert cfg.model.n_head == Config().model.n_head  # untouched keys keep defaults


def test_unknown_key_rejected():
    with pytest.raises(KeyError, match="n_layers"):
        config_from_dict({"model": {"n_layers": 6}})


def test_unknown_section_rejected():
    with pytest.raises(KeyError, match="modle"):
        config_from_dict({"modle": {}})


@pytest.mark.parametrize("section, key, value", [
    ("model", "n_layer", "four"),
    ("model", "n_layer", 4.5),
    ("model", "n_layer", True),   # bool is an int subclass in Python; still rejected
    ("train", "lr", "fast"),
    ("model", "bias", "yes"),
])
def test_wrong_type_rejected(section, key, value):
    with pytest.raises(TypeError):
        config_from_dict({section: {key: value}})


def test_yaml_scientific_notation_is_float(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 3e-4\n")  # PyYAML reads this as the string "3e-4"
    assert load_config(path).train.lr == 3e-4


def test_compile_accepts_bool_or_auto():
    assert config_from_dict({"system": {"compile": False}}).system.compile is False
    assert config_from_dict({"system": {"compile": "auto"}}).system.compile == "auto"


def test_model_validation():
    with pytest.raises(ValueError, match="divisible"):
        config_from_dict({"model": {"n_embd": 130, "n_head": 4}})
    with pytest.raises(ValueError, match="pos_encoding"):
        config_from_dict({"model": {"pos_encoding": "alibi"}})


def test_command_line_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  n_layer: 4\n")
    cfg = load_config(path, ["model.n_layer=8", "model.pos_encoding=learned",
                             "train.lr=3e-4", "system.compile=false"])
    assert cfg.model.n_layer == 8
    assert cfg.model.pos_encoding == "learned"
    assert cfg.train.lr == 3e-4
    assert cfg.system.compile is False


def test_round_trip_through_dict():
    cfg = config_from_dict({"model": {"n_layer": 6}, "train": {"lr": 3e-4}})
    assert config_from_dict(cfg.to_dict()) == cfg  # how checkpoints will restore the config
