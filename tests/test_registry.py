"""
Unit tests for sqst.registry.
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from huggingface_hub import constants as hf_constants

from sqst.registry import (
    get_model_info,
    is_cached_locally,
    list_models,
    resolve_model_id,
)


def test_list_models():
    models = list_models()
    assert len(models) == 7
    repos = [m.repo_id for m in models]
    assert "Squeal-Studio/squeal_ai_8m-instruct" in repos
    assert "Squeal-Studio/squeal_ai_20m-instruct" in repos
    assert "Squeal-Studio/drakon_28m-instruct" in repos
    assert "Squeal-Studio/squeal_lstm_5m" in repos


@pytest.mark.parametrize(
    "alias,expected_repo",
    [
        ("8m", "Squeal-Studio/squeal_ai_8m-instruct"),
        ("8m-instruct", "Squeal-Studio/squeal_ai_8m-instruct"),
        ("squeal-8m", "Squeal-Studio/squeal_ai_8m-instruct"),
        ("8m-base", "Squeal-Studio/squeal_ai_8m-base"),
        ("20m", "Squeal-Studio/squeal_ai_20m-instruct"),
        ("20m-instruct", "Squeal-Studio/squeal_ai_20m-instruct"),
        ("20m-base", "Squeal-Studio/squeal_ai_20m-base"),
        ("lstm", "Squeal-Studio/squeal_lstm_5m"),
        ("lstm-5m", "Squeal-Studio/squeal_lstm_5m"),
        ("squeal_lstm", "Squeal-Studio/squeal_lstm_5m"),
        ("drakon", "Squeal-Studio/drakon_28m-instruct"),
        ("drakon-instruct", "Squeal-Studio/drakon_28m-instruct"),
        ("drakon-base", "Squeal-Studio/drakon_28m-base"),
        ("  DRAKON  ", "Squeal-Studio/drakon_28m-instruct"),
    ],
)
def test_resolve_model_id_aliases(alias, expected_repo):
    assert resolve_model_id(alias) == expected_repo
    info = get_model_info(alias)
    assert info is not None
    assert info.repo_id == expected_repo


def test_resolve_custom_repo():
    custom = "custom-user/my-custom-model"
    assert resolve_model_id(custom) == custom
    assert get_model_info(custom) is None


def test_resolve_empty_name_raises():
    with pytest.raises(ValueError):
        resolve_model_id("   ")


def test_path_like_names_are_never_treated_as_aliases():
    assert resolve_model_id("./8m") == "./8m"
    assert resolve_model_id("/models/drakon") == "/models/drakon"


def _fake_snapshot(cache: Path, repo: str, filenames):
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in filenames:
        (snap / name).write_text("x")


def test_is_cached_locally_detects_single_and_sharded_weights():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        with mock.patch.object(hf_constants, "HF_HUB_CACHE", str(cache)):
            assert not is_cached_locally("8m")

            _fake_snapshot(cache, "Squeal-Studio/squeal_ai_8m-instruct", ["model.safetensors"])
            assert is_cached_locally("8m")

            _fake_snapshot(cache, "Squeal-Studio/squeal_ai_20m-instruct",
                           ["model.safetensors.index.json"])
            assert is_cached_locally("20m")

            _fake_snapshot(cache, "Squeal-Studio/drakon_28m-instruct", ["config.json"])
            assert not is_cached_locally("drakon")


def test_is_cached_locally_for_local_directory():
    with tempfile.TemporaryDirectory() as tmp:
        if not tmp.startswith("/"):
            pytest.skip("absolute-path check is POSIX-specific")
        assert is_cached_locally(tmp)
        assert not is_cached_locally(tmp + "/missing")
