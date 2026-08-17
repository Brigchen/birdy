# -*- coding: utf-8 -*-
"""动图弹窗选项参数：打开时恢复上次保存的值。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).resolve().parent / "burst_dialog_state_helper.py"


def _run_helper(cmd: str, tmp_path: Path) -> None:
    log = tmp_path / f"{cmd}.log"
    with log.open("w", encoding="utf-8") as fh:
        r = subprocess.run(
            [sys.executable, str(HELPER), cmd, str(tmp_path)],
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if r.returncode != 0:
        detail = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        pytest.fail(f"helper {cmd} exited {r.returncode}\n{detail}")


def test_burst_dialog_options_roundtrip(tmp_path):
    _run_helper("roundtrip", tmp_path)


def test_burst_dialog_init_does_not_clobber_saved_options(tmp_path):
    _run_helper("noclobber", tmp_path)


def test_burst_dialog_project_roundtrip(tmp_path):
    _run_helper("project", tmp_path)
